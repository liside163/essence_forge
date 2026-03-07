"""
datasets.py

PyTorch 数据集定义。

本模块同时支持两种采样模式：
1. 传统多尺度随机切窗（兼容旧实验）。
2. S6 阶段切窗（按故障注入阶段重标注 + 健康掩码拼接）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import sys


from essence_forge.core.runtime_config import CFG
from essence_forge.core.channel_layout import (
    LEGACY_ACCEL_INDICES,
    LEGACY_CORE_CHANNEL_COUNT,
    LEGACY_POS_INDICES,
    LEGACY_VEL_INDICES,
)
from essence_forge.core.gan_augment import (
    GanTrainingConfig,
    TargetedSyntheticBankAugmentor,
    build_synthetic_bank_cache_key,
    build_class_difficulty_from_eval_json,
    load_or_train_synthetic_bank,
    select_hard_classes_from_eval_json,
    train_targeted_wgan_synthetic_bank,
)
from essence_forge.core.rflymad_io import MissionLoader
from essence_forge.core.utils import stable_seed_from_items

# 物理感知增强器（P1+P2+P3）
from essence_forge.core.augment import (
    AugContext,
    PhysicsAwareAugmentor,
    FreqDomainAugmentor,
    HardClassMixupAugmentor,
    HardClassMixupConfig,
    AugRollbackController,
    AugPolicy,
    POLICIES,
    RiskMetrics,
    build_augmentor_from_config,
    validate_physics_constraints,
)


@dataclass
class Sample:
    """
    单个样本。

    维度说明：
    - `x`: `[T, C]`
    - `length`: 有效长度（不含 padding）
    - `y`: 标量类别 ID
    """

    x: torch.Tensor
    length: int
    y: torch.Tensor


@dataclass(frozen=True)
class StageWindowRecord:
    """
    阶段切窗索引条目。

    字段：
    - `record_idx`: 对应 mission 在 records 中的下标。
    - `start`: 窗口起点（时间轴下标）。
    - `window_length`: 固定窗口长度。
    - `class_id`: 当前窗口标签。
    """

    record_idx: int
    start: int
    window_length: int
    class_id: int


def compute_fault_onset_idx(fault_state: np.ndarray) -> int:
    """
    计算故障注入起点：首次非零索引；无故障返回 -1。
    """

    non_zero = np.where(np.asarray(fault_state).reshape(-1) != 0)[0]
    if len(non_zero) == 0:
        return -1
    return int(non_zero[0])


def build_stage_windows(
    total_length: int,
    fault_code: int,
    fault_onset_idx: int,
    window_len: int,
    stride: int,
    normal_class_id: int = 10,
    drop_prefault_normal_windows_for_fault_missions: bool = False,
) -> List[Tuple[int, int, int]]:
    """
    生成阶段切窗列表。

    输出每个元素 `(start, end, class_id)`，标签策略固定为：
    - `fault_code == 10`：全窗口标注 `normal_class_id`。
    - `fault_code in [0..9]`：
      - `start < onset` => `normal_class_id` 或直接丢弃（当启用预故障窗口丢弃开关）
      - `start >= onset` => `fault_code`
    """

    if total_length <= 0 or window_len <= 0 or stride <= 0:
        return []

    if total_length >= window_len:
        max_start = total_length - window_len
        starts = list(range(0, max_start + 1, stride))
    else:
        # 序列短于窗口时仍保留一个窗口，由上层负责 padding。
        starts = [0]

    windows: List[Tuple[int, int, int]] = []
    onset = int(fault_onset_idx)
    if onset < 0:
        onset = 0

    for start in starts:
        end = start + window_len
        if int(fault_code) == 10:
            class_id = int(normal_class_id)
        else:
            if drop_prefault_normal_windows_for_fault_missions and start < onset:
                continue
            class_id = int(normal_class_id) if start < onset else int(fault_code)
        windows.append((int(start), int(end), int(class_id)))
    return windows


def generate_health_mask(
    total_length: int,
    fault_code: int,
    fault_onset_idx: int,
    channel_names: Sequence[str],
) -> np.ndarray:
    """
    生成健康掩码 `m_health`，维度 `[T, C]`。

    规则：
    - 默认全 1。
    - 仅对四类传感器故障注入后置 0：
      - 5 -> accel_x/y/z
      - 6 -> gyro_x/y/z
      - 7 -> mag_x/y/z
      - 8 -> baro_alt/baro_temp/baro_pressure
    """

    T = int(max(total_length, 0))
    C = int(len(channel_names))
    mask = np.ones((T, C), dtype=np.float32)

    fault_to_channels = {
        5: ("accel_x", "accel_y", "accel_z"),
        6: ("gyro_x", "gyro_y", "gyro_z"),
        7: ("mag_x", "mag_y", "mag_z"),
        8: ("baro_alt", "baro_temp", "baro_pressure"),
    }
    target_channels = fault_to_channels.get(int(fault_code), ())
    onset = int(fault_onset_idx)
    if onset < 0 or len(target_channels) == 0:
        return mask

    name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
    channel_indices = [name_to_idx[name] for name in target_channels if name in name_to_idx]
    if len(channel_indices) == 0:
        return mask

    onset = min(max(onset, 0), T)
    mask[onset:, channel_indices] = 0.0
    return mask


def _normalize_window_with_stats(
    raw_window: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    使用统计量对窗口做 z-score 标准化。

    兼容统计量长度大于原始通道数的场景（例如未来扩展到 28ch 统计量），
    仅取前 `C_raw` 个通道统计量参与归一化。
    """

    if raw_window.ndim != 2:
        raise ValueError(f"raw_window 必须是 2D，实际={raw_window.ndim}D")

    c_raw = int(raw_window.shape[1])
    mean_vec = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_vec = np.maximum(np.asarray(std, dtype=np.float32).reshape(-1), 1e-8)
    if mean_vec.shape[0] < c_raw or std_vec.shape[0] < c_raw:
        raise ValueError(
            "z-score 统计量维度不足："
            f"raw_channels={c_raw}, mean={mean_vec.shape[0]}, std={std_vec.shape[0]}"
        )
    return ((raw_window - mean_vec[:c_raw]) / std_vec[:c_raw]).astype(np.float32)


def compute_cross_sensor_residuals(
    raw: np.ndarray,
    dt: float = 1.0 / 120.0,
    normalize: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    从任意 `C >= 19` 的原始数据计算 9ch 交叉传感器一致性残差。

    输入: raw [T, C] (未 z-score, C >= 19)
    输出: residuals [T, 9]

    仅使用 legacy core 前 19 通道中的 accel/pos/vel 位置。
    """

    raw_arr = np.asarray(raw, dtype=np.float32)
    if raw_arr.ndim != 2:
        raise ValueError(f"raw 必须是 2D 数组，实际={raw_arr.ndim}D")
    if raw_arr.shape[1] < LEGACY_CORE_CHANNEL_COUNT:
        raise ValueError(
            f"raw 通道数不足，至少需要 {LEGACY_CORE_CHANNEL_COUNT}，实际={raw_arr.shape[1]}"
        )
    if float(dt) <= 0.0:
        raise ValueError(f"dt 必须 > 0，当前={dt}")

    accel = raw_arr[:, list(LEGACY_ACCEL_INDICES)]   # [T, 3]
    vel = raw_arr[:, list(LEGACY_VEL_INDICES)]       # [T, 3]
    pos = raw_arr[:, list(LEGACY_POS_INDICES)]       # [T, 3]

    residuals = np.zeros((raw_arr.shape[0], 9), dtype=np.float32)

    # ch 0-2: IMU-EKF 速度一致性, d(vel)/dt ≈ accel
    dvel = np.diff(vel, axis=0, prepend=vel[:1]) / float(dt)
    residuals[:, 0:3] = dvel - accel

    # ch 3-5: 位置-速度一致性, d(pos)/dt ≈ vel
    dpos = np.diff(pos, axis=0, prepend=pos[:1]) / float(dt)
    residuals[:, 3:6] = dpos - vel

    # ch 6-8: 加速度-位置二阶一致性, d²(pos)/dt² ≈ accel
    ddpos = np.diff(dpos, axis=0, prepend=dpos[:1]) / float(dt)
    residuals[:, 6:9] = ddpos - accel

    if bool(normalize):
        scale = np.std(residuals, axis=0, dtype=np.float32)
        scale = np.maximum(scale, float(eps)).astype(np.float32)
        residuals = residuals / scale.reshape(1, -1)

    return residuals.astype(np.float32, copy=False)


def _append_cross_sensor_residual_features(
    normalized_window: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    基于标准化后的基准通道重建物理量并追加残差通道。

    设计原因：
    1. 当前增强流程在 z-score 后进行；先增强再反标准化可保持残差与增强后输入一致。
    2. 残差采用窗口级标准差归一化（方案 B），无需依赖 28ch 全局统计量。
    """

    if normalized_window.ndim != 2:
        raise ValueError(f"normalized_window 必须是 2D，实际={normalized_window.ndim}D")

    base_channels = int(getattr(CFG, "input_dim", normalized_window.shape[1]))
    if normalized_window.shape[1] < base_channels:
        raise ValueError(
            f"窗口通道数小于基础通道数，window={normalized_window.shape[1]}, base={base_channels}"
        )

    residual_channels = int(getattr(CFG, "cross_sensor_residual_channels", 9))
    if residual_channels != 9:
        raise ValueError(
            f"当前实现固定追加 9 个残差通道，配置为 {residual_channels} 不受支持"
        )

    mean_vec = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_vec = np.maximum(np.asarray(std, dtype=np.float32).reshape(-1), 1e-8)
    if mean_vec.shape[0] < base_channels or std_vec.shape[0] < base_channels:
        raise ValueError(
            "基础通道反标准化失败："
            f"base={base_channels}, mean={mean_vec.shape[0]}, std={std_vec.shape[0]}"
        )

    base_norm = normalized_window[:, :base_channels].astype(np.float32, copy=False)
    base_phys = base_norm * std_vec[:base_channels].reshape(1, -1) + mean_vec[:base_channels].reshape(1, -1)

    sample_rate_hz = max(float(getattr(CFG, "windows_sample_rate_hz", 120)), 1e-6)
    residuals = compute_cross_sensor_residuals(
        base_phys,
        dt=1.0 / sample_rate_hz,
        normalize=bool(getattr(CFG, "cross_sensor_residual_window_norm", True)),
        eps=float(getattr(CFG, "cross_sensor_residual_norm_eps", 1e-6)),
    )
    return np.concatenate([normalized_window, residuals], axis=1).astype(np.float32, copy=False)


class SourceDomainDataset(Dataset):
    """
    源域训练集。

    支持：
    - 传统多尺度随机切窗。
    - 阶段切窗 + 健康掩码拼接（S6）。
    """

    def __init__(
        self,
        records: Sequence[Dict],
        zscore_mean: np.ndarray,
        zscore_std: np.ndarray,
        loader: MissionLoader,
        is_train: bool = True,
        base_seed: int = 42,
        enable_targeted_gan: bool = False,
        enable_physics_augment: bool = True,
        augment_mode: str = "full",
    ):
        self.records = list(records)
        self.mean = np.asarray(zscore_mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(zscore_std, dtype=np.float32).reshape(-1)
        self.std = np.maximum(self.std, 1e-8)
        self.loader = loader
        self.is_train = bool(is_train)
        self.base_seed = int(base_seed)
        self._epoch = 0
        self.channel_names = tuple(ch.name for ch in CFG.channels)
        self._health_mask_cache: Dict[str, np.ndarray] = {}
        self.use_stage_windowing = bool(getattr(CFG, "windows_enable_stage_windowing", False))
        self.enable_targeted_gan = bool(enable_targeted_gan and self.is_train)
        self.targeted_gan_augmentor: Optional[TargetedSyntheticBankAugmentor] = None

        # 物理感知增强器（P1+P2+P3）
        self.enable_physics_augment = bool(enable_physics_augment and self.is_train)
        self.augment_mode = str(augment_mode).strip().lower()
        if not self.is_train:
            self.augment_mode = "off"
        if self.augment_mode not in {"full", "light", "off"}:
            raise ValueError("augment_mode 必须是 full/light/off")
        self.physics_augmentor: Optional[PhysicsAwareAugmentor] = None
        self.hard_class_mixup: Optional[HardClassMixupAugmentor] = None

        if self.enable_physics_augment:
            self._init_physics_augmentor()

        if self.use_stage_windowing:
            self.stage_window_len = int(round(CFG.windows_stage_window_seconds * CFG.windows_sample_rate_hz))
            self.stage_stride = int(round(CFG.windows_stage_stride_seconds * CFG.windows_sample_rate_hz))
            if self.stage_window_len <= 0 or self.stage_stride <= 0:
                raise ValueError("阶段窗口长度和步长必须大于 0")
            self.stage_records = self._build_stage_window_records()
            self.stage_records = self._maybe_downsample_normal_stage_windows(self.stage_records)
            self._total_len = len(self.stage_records)
            if self.enable_targeted_gan:
                self.targeted_gan_augmentor = self._build_targeted_gan_augmentor()
        else:
            self.windows_per_scale = int(CFG.train_windows_per_mission_per_scale)
            self.num_scales = len(CFG.window_lengths)
            self.samples_per_mission = self.windows_per_scale * self.num_scales
            self._total_len = len(self.records) * self.samples_per_mission

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，用于训练期增强随机数。"""

        if self.hard_class_mixup is not None and int(epoch) != self._epoch:
            self.hard_class_mixup.clear_bank()
        self._epoch = int(epoch)

    def _init_physics_augmentor(self) -> None:
        """
        初始化物理感知增强器。

        从 CFG 读取配置并构建增强器实例。
        """
        if not self.is_train:
            return

        if self.augment_mode == "off":
            print("[增强] augment_mode=off，训练期增强已关闭")
            return

        if self.augment_mode == "light":
            print("[增强] augment_mode=light，使用轻量扰动增强")
            return

        # 检查是否启用物理感知增强
        strategy = str(getattr(CFG, "augment_strategy", "legacy")).lower()
        if strategy != "physics_aware":
            print("[增强] 使用传统增强策略（legacy）")
            return

        # 构建增强器
        self.physics_augmentor = build_augmentor_from_config(CFG)
        print("[增强] 物理感知增强器已初始化")

        # 硬类 Mixup
        if self.augment_mode == "full" and bool(
            getattr(
                CFG,
                "augment_class_mixup_enable",
                getattr(CFG, "augment_hard_class_mixup_enable", False),
            )
        ):
            self.hard_class_mixup = HardClassMixupAugmentor(
                cfg=HardClassMixupConfig(
                    enable=True,
                    prob=float(
                        getattr(
                            CFG,
                            "augment_class_mixup_prob",
                            getattr(CFG, "augment_hard_class_mixup_prob", 0.35),
                        )
                    ),
                    alpha=float(
                        getattr(
                            CFG,
                            "augment_class_mixup_alpha",
                            getattr(CFG, "augment_hard_class_mixup_alpha", 0.3),
                        )
                    ),
                    target_classes=tuple(
                        int(v)
                        for v in getattr(
                            CFG,
                            "augment_class_mixup_target_classes",
                            getattr(CFG, "augment_hard_class_targets", (5, 6, 7)),
                        )
                    ),
                    same_fault_family_only=bool(
                        getattr(CFG, "augment_class_mixup_same_fault_family_only", False)
                    ),
                    max_bank_size_per_class=int(
                        getattr(CFG, "augment_class_mixup_max_bank_size_per_class", 128)
                    ),
                )
            )
            print("[增强] 动态类别 Mixup 已启用")

    def set_augmentor_policy(self, policy: "AugPolicy") -> None:
        """
        设置当前增强策略（用于回退机制）。

        由训练循环调用，根据监控指标动态调整。
        """
        if self.physics_augmentor is not None:
            self.physics_augmentor.set_policy(policy)

    def update_class_recall(self, class_recall: Dict[int, float]) -> None:
        """
        更新各类召回率（用于动态调整增强概率）。

        由训练循环在每个 epoch 结束后调用。
        """
        if self.physics_augmentor is not None:
            self.physics_augmentor.update_class_recall(class_recall)

    def __len__(self) -> int:
        return self._total_len

    def _rng_for_index(self, index: int) -> np.random.Generator:
        """
        为样本索引构造可复现随机数生成器。
        """

        # 使用稳定哈希替代内置 hash()，避免跨进程随机盐破坏复现性。
        seed = stable_seed_from_items(self.base_seed, self._epoch, int(index))
        return np.random.default_rng(seed)

    def _build_stage_window_records(self) -> List[StageWindowRecord]:
        """
        构建阶段窗口索引。

        维度变化（概念）：
        - mission 序列: `[T, C]`
        - 按步长切成多个窗口：每个窗口 `[window_len, C]`
        - 每个窗口绑定独立标签（预故障=10，注入后=故障类）
        """

        records: List[StageWindowRecord] = []
        normal_class_id = int(CFG.fault_to_class.get(10, 10))

        for record_idx, record in enumerate(self.records):
            file_path = record["file_path"]
            data = self.loader.load(file_path)
            T = int(len(data))
            fault_code = int(record.get("fault_code", 10))

            meta = self.loader.load_metadata(file_path)
            onset_idx = int(meta.fault_onset_idx)
            if fault_code != 10 and onset_idx < 0:
                onset_idx = int(round(CFG.windows_normal_context_seconds * CFG.windows_sample_rate_hz))

            windows = build_stage_windows(
                total_length=T,
                fault_code=fault_code,
                fault_onset_idx=onset_idx,
                window_len=self.stage_window_len,
                stride=self.stage_stride,
                normal_class_id=normal_class_id,
                drop_prefault_normal_windows_for_fault_missions=bool(
                    getattr(
                        CFG,
                        "windows_drop_prefault_normal_windows_for_fault_missions",
                        False,
                    )
                ),
            )
            for start, _, label_fault_or_normal in windows:
                class_id = int(CFG.fault_to_class.get(int(label_fault_or_normal), int(label_fault_or_normal)))
                records.append(
                    StageWindowRecord(
                        record_idx=record_idx,
                        start=int(start),
                        window_length=int(self.stage_window_len),
                        class_id=class_id,
                    )
                )
        return records

    def _maybe_downsample_normal_stage_windows(
        self,
        stage_records: List[StageWindowRecord],
    ) -> List[StageWindowRecord]:
        """
        可选下采样训练集中的正常类窗口，缓解极端不平衡。
        """

        if not self.is_train:
            return stage_records

        ratio = float(getattr(CFG, "normal_class_downsample_ratio", 1.0))
        if ratio >= 1.0:
            return stage_records

        normal_class_id = int(CFG.fault_to_class.get(10, 10))
        rng = np.random.default_rng(self.base_seed)
        kept: List[StageWindowRecord] = []
        for item in stage_records:
            if item.class_id != normal_class_id:
                kept.append(item)
                continue
            if rng.random() <= ratio:
                kept.append(item)
        return kept

    def _collect_stage_windows_by_class(
        self,
        class_ids: Sequence[int],
        max_windows_per_class: int,
    ) -> Dict[int, np.ndarray]:
        """
        从阶段窗口索引中收集指定类别窗口（标准化后）。
        """

        wanted = {int(x) for x in class_ids}
        if len(wanted) == 0:
            return {}

        idx_by_class: Dict[int, List[int]] = {cid: [] for cid in wanted}
        for idx, item in enumerate(self.stage_records):
            if item.class_id in idx_by_class:
                idx_by_class[item.class_id].append(idx)

        rng = np.random.default_rng(self.base_seed + 1337)
        out: Dict[int, np.ndarray] = {}
        for class_id, indices in idx_by_class.items():
            if len(indices) == 0:
                continue
            rng.shuffle(indices)
            selected = indices[: int(max_windows_per_class)]
            windows: List[np.ndarray] = []
            for stage_idx in selected:
                spec = self.stage_records[stage_idx]
                record = self.records[spec.record_idx]
                data = self.loader.load(record["file_path"])
                raw_window, _ = self._slice_with_pad(
                    array=data,
                    start=spec.start,
                    window_length=spec.window_length,
                    pad_value=0.0,
                )
                raw_window = _normalize_window_with_stats(
                    raw_window=raw_window,
                    mean=self.mean,
                    std=self.std,
                )
                windows.append(raw_window)

            if len(windows) > 0:
                out[int(class_id)] = np.stack(windows, axis=0).astype(np.float32)
        return out

    def _build_targeted_gan_augmentor(self) -> Optional[TargetedSyntheticBankAugmentor]:
        """
        构建 hard-class 定向 GAN 增强器（仅训练期生效）。
        """

        if not bool(getattr(CFG, "targeted_gan_enable", False)):
            return None
        if not self.use_stage_windowing:
            print("[GAN增强] 当前仅在阶段切窗模式启用，已跳过。")
            return None

        eval_json = str(getattr(CFG, "targeted_gan_reference_eval_json", "")).strip()
        if len(eval_json) == 0:
            print("[GAN增强] 未配置 reference_eval_json，已跳过。")
            return None

        manual_hard_classes = tuple(int(x) for x in getattr(CFG, "targeted_gan_manual_hard_classes", ()))
        if len(manual_hard_classes) > 0:
            hard_classes = list(manual_hard_classes)
        else:
            hard_classes = select_hard_classes_from_eval_json(
                eval_json_path=eval_json,
                top_k=int(getattr(CFG, "targeted_gan_top_k", 5)),
                min_support=int(getattr(CFG, "targeted_gan_min_support", 30)),
                exclude_classes=tuple(int(x) for x in getattr(CFG, "targeted_gan_exclude_classes", (10,))),
            )

        if len(hard_classes) == 0:
            print("[GAN增强] 没有可增强的 hard classes，已跳过。")
            return None

        difficulty_rows = build_class_difficulty_from_eval_json(
            eval_json_path=eval_json,
            exclude_classes=tuple(int(x) for x in getattr(CFG, "targeted_gan_exclude_classes", (10,))),
        )
        difficulty_map = {row.class_id: row for row in difficulty_rows}
        hard_descriptions = []
        for class_id in hard_classes:
            name = str(CFG.class_id_to_name.get(int(class_id), f"class_{class_id}"))
            row = difficulty_map.get(int(class_id))
            if row is None:
                hard_descriptions.append(f"{class_id}({name})")
            else:
                hard_descriptions.append(
                    f"{class_id}({name}, recall={row.recall:.3f}, support={row.support})"
                )
        print(f"[GAN增强] hard classes: {', '.join(hard_descriptions)}")

        class_windows = self._collect_stage_windows_by_class(
            class_ids=hard_classes,
            max_windows_per_class=int(getattr(CFG, "targeted_gan_max_windows_per_class", 512)),
        )
        if len(class_windows) == 0:
            print("[GAN增强] 训练集未收集到 hard class 窗口，已跳过。")
            return None

        train_cfg = GanTrainingConfig(
            latent_dim=int(getattr(CFG, "targeted_gan_latent_dim", 96)),
            hidden_dim=int(getattr(CFG, "targeted_gan_hidden_dim", 256)),
            train_steps=int(getattr(CFG, "targeted_gan_train_steps", 180)),
            critic_steps=int(getattr(CFG, "targeted_gan_critic_steps", 3)),
            batch_size=int(getattr(CFG, "targeted_gan_batch_size", 64)),
            lr_generator=float(getattr(CFG, "targeted_gan_lr_generator", 2e-4)),
            lr_discriminator=float(getattr(CFG, "targeted_gan_lr_discriminator", 4e-4)),
            gp_lambda=float(getattr(CFG, "targeted_gan_gp_lambda", 10.0)),
            spectral_lambda=float(getattr(CFG, "targeted_gan_spectral_lambda", 1.0)),
            smooth_lambda=float(getattr(CFG, "targeted_gan_smooth_lambda", 0.02)),
            synth_bank_size=int(getattr(CFG, "targeted_gan_synth_bank_size", 512)),
            min_windows_per_class=int(getattr(CFG, "targeted_gan_min_windows_per_class", 24)),
            seed=int(getattr(CFG, "targeted_gan_seed", self.base_seed)),
            device=str(getattr(CFG, "targeted_gan_device", "cpu")),
        )

        cache_enable = bool(getattr(CFG, "targeted_gan_cache_enable", False))
        cache_force_retrain = bool(getattr(CFG, "targeted_gan_cache_force_retrain", False))
        cache_dir = str(getattr(CFG, "targeted_gan_cache_dir", "")).strip()
        cache_path: Optional[Path] = None
        if cache_enable and len(cache_dir) > 0:
            cache_key = build_synthetic_bank_cache_key(
                hard_classes=hard_classes,
                class_windows=class_windows,
                config=train_cfg,
            )
            cache_path = Path(cache_dir).expanduser().resolve() / f"targeted_bank_{cache_key}.npz"

        if cache_path is not None:
            synth_bank, train_stats, cache_hit = load_or_train_synthetic_bank(
                cache_path=cache_path,
                class_windows=class_windows,
                config=train_cfg,
                force_retrain=cache_force_retrain,
            )
            print(
                f"[GAN增强] synthetic bank 缓存{'命中' if cache_hit else '回源训练'}: {cache_path}"
            )
        else:
            synth_bank, train_stats = train_targeted_wgan_synthetic_bank(
                class_windows=class_windows,
                config=train_cfg,
            )

        if len(synth_bank) == 0:
            print("[GAN增强] GAN 训练后未生成可用 synthetic bank，已跳过。")
            return None

        for class_id, stats in train_stats.items():
            name = str(CFG.class_id_to_name.get(int(class_id), f"class_{class_id}"))
            print(
                "[GAN增强] "
                f"class={class_id}({name}) "
                f"train_windows={int(stats['train_windows'])} "
                f"synth_windows={int(stats['synthetic_windows'])} "
                f"last_d_loss={stats['last_d_loss']:.4f} "
                f"last_g_loss={stats['last_g_loss']:.4f}"
            )

        return TargetedSyntheticBankAugmentor(
            synthetic_bank_by_class=synth_bank,
            apply_prob=float(getattr(CFG, "targeted_gan_apply_prob", 0.8)),
            blend_ratio_min=float(getattr(CFG, "targeted_gan_blend_ratio_min", 0.35)),
            blend_ratio_max=float(getattr(CFG, "targeted_gan_blend_ratio_max", 0.65)),
        )

    @staticmethod
    def _slice_with_pad(
        array: np.ndarray,
        start: int,
        window_length: int,
        pad_value: float,
    ) -> Tuple[np.ndarray, int]:
        """
        从时间序列切窗并在末尾补齐。

        输入维度：
        - `array`: `[T, C]`
        输出维度：
        - `window`: `[window_length, C]`
        """

        T = int(array.shape[0])
        C = int(array.shape[1])
        start = max(int(start), 0)
        end = start + int(window_length)

        out = np.full((window_length, C), pad_value, dtype=np.float32)
        if start >= T:
            return out, 0

        real_end = min(end, T)
        real_len = max(real_end - start, 0)
        if real_len > 0:
            out[:real_len] = array[start:real_end]
        return out, int(real_len)

    def _get_health_mask_full(self, record: Dict, total_length: int) -> np.ndarray:
        """
        获取 mission 级健康掩码，维度 `[T, C]`。
        """

        file_path = record["file_path"]
        if file_path in self._health_mask_cache:
            return self._health_mask_cache[file_path]

        fault_code = int(record.get("fault_code", 10))
        meta = self.loader.load_metadata(file_path)
        onset_idx = int(meta.fault_onset_idx)
        if fault_code != 10 and onset_idx < 0:
            onset_idx = int(round(CFG.windows_normal_context_seconds * CFG.windows_sample_rate_hz))

        mask = generate_health_mask(
            total_length=total_length,
            fault_code=fault_code,
            fault_onset_idx=onset_idx,
            channel_names=self.channel_names,
        )
        self._health_mask_cache[file_path] = mask
        return mask

    def _apply_augment(
        self,
        x: np.ndarray,
        y: int,
        rng: np.random.Generator,
        epoch: int = 0,
        context: Optional[AugContext] = None,
    ) -> np.ndarray:
        """
        数据增强（仅作用于原始传感器特征，不作用于健康掩码）。

        支持两种模式:
        1. 物理感知增强（strategy="physics_aware"）: 使用 PhysicsAwareAugmentor
        2. 传统增强（fallback）: 简单高斯噪声 + 线性漂移
        """
        if self.augment_mode == "off":
            return x

        if self.augment_mode == "light":
            sigma = max(float(CFG.augment_gaussian_sigma) * 0.5, 0.0)
            if sigma > 0:
                noise = rng.normal(0.0, sigma, x.shape)
                x = x + noise.astype(np.float32)

            if CFG.enable_drift_noise and CFG.augment_drift_sigma > 0:
                T, C = x.shape
                slopes = rng.normal(0.0, float(CFG.augment_drift_sigma) * 0.25, (1, C))
                t_normalized = np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(-1, 1)
                drift = (slopes * t_normalized).astype(np.float32)
                x = x + drift
            return x

        # 优先使用物理感知增强器
        if self.physics_augmentor is not None:
            return self.physics_augmentor.augment(x, y, epoch, rng, context=context)

        # Fallback: 传统简单增强
        if CFG.augment_gaussian_sigma > 0:
            noise = rng.normal(0.0, float(CFG.augment_gaussian_sigma), x.shape)
            x = x + noise.astype(np.float32)

        if CFG.enable_drift_noise and CFG.augment_drift_sigma > 0:
            T, C = x.shape
            slopes = rng.normal(0.0, float(CFG.augment_drift_sigma), (1, C))
            t_normalized = np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(-1, 1)
            drift = (slopes * t_normalized).astype(np.float32)
            x = x + drift

        return x

    def _build_sample(
        self,
        record: Dict,
        start: int,
        window_length: int,
        class_id: int,
        rng: np.random.Generator,
    ) -> Sample:
        """
        构建单个训练样本。

        维度路径：
        - 基础原始窗口 `x_raw`: `[T, C_base]`（例如当前主配置 `C_base=30`）
        - 可选残差窗口 `x_res`: `[T, 9]`
        - 健康掩码 `m_health`: `[T, C_base]`
        - 拼接后输入：
          - 仅基准通道 + 掩码：`[T, C_base + C_base]`（默认 60）
          - 启用残差后：`[T, C_base + 9 + C_base]`（默认 69）
        """

        file_path = record["file_path"]
        data = self.loader.load(file_path)  # [T, C_raw]
        raw_window, real_len = self._slice_with_pad(
            array=data,
            start=start,
            window_length=window_length,
            pad_value=0.0,
        )
        raw_window = _normalize_window_with_stats(
            raw_window=raw_window,
            mean=self.mean,
            std=self.std,
        )

        if self.is_train and self.targeted_gan_augmentor is not None:
            raw_window = self.targeted_gan_augmentor.maybe_augment(
                class_id=int(class_id),
                real_window=raw_window,
                rng=rng,
            )

        aug_context: Optional[AugContext] = None
        legacy_core_channels = min(LEGACY_CORE_CHANNEL_COUNT, int(raw_window.shape[1]))
        raw_core = raw_window[:, :legacy_core_channels].copy()
        raw_tail = raw_window[:, legacy_core_channels:].copy()
        if self.is_train and CFG.enable_augment:
            fault_code = int(record.get("fault_code", 10))
            meta = self.loader.load_metadata(file_path)
            onset_idx = int(meta.fault_onset_idx)
            if fault_code != 10 and onset_idx < 0:
                onset_idx = int(round(CFG.windows_normal_context_seconds * CFG.windows_sample_rate_hz))
            aug_context = AugContext(
                fault_code=fault_code,
                fault_onset_idx=onset_idx,
                window_start=int(start),
                window_length=int(window_length),
                channel_names=tuple(self.channel_names[:legacy_core_channels]),
            )
            raw_core = self._apply_augment(
                raw_core,
                y=int(class_id),
                rng=rng,
                epoch=self._epoch,
                context=aug_context,
            )

        # P3: 硬类 Mixup（可选）
        if self.is_train and self.hard_class_mixup is not None:
            self.hard_class_mixup.register_sample(y=int(class_id), x=raw_core)
            raw_core = self.hard_class_mixup.maybe_mixup(
                x=raw_core,
                y=int(class_id),
                rng=rng,
            )

        raw_window = np.concatenate([raw_core, raw_tail], axis=1).astype(
            np.float32,
            copy=False,
        )

        if bool(getattr(CFG, "use_cross_sensor_residuals", False)):
            raw_window = _append_cross_sensor_residual_features(
                normalized_window=raw_window,
                mean=self.mean,
                std=self.std,
            )

        if bool(getattr(CFG, "concat_health_mask_channels", True)):
            health_full = self._get_health_mask_full(record, total_length=int(len(data)))
            health_window, _ = self._slice_with_pad(
                array=health_full,
                start=start,
                window_length=window_length,
                pad_value=1.0,
            )
            x_np = np.concatenate([raw_window, health_window.astype(np.float32)], axis=1)
        else:
            x_np = raw_window

        x = torch.from_numpy(x_np.astype(np.float32))
        y = torch.tensor(int(class_id), dtype=torch.long)
        return Sample(x=x, length=int(real_len), y=y)

    def __getitem__(self, index: int) -> Sample:
        rng = self._rng_for_index(index)

        if self.use_stage_windowing:
            spec = self.stage_records[index]
            record = self.records[spec.record_idx]
            return self._build_sample(
                record=record,
                start=spec.start,
                window_length=spec.window_length,
                class_id=spec.class_id,
                rng=rng,
            )

        # 传统模式：随机多尺度切窗
        mission_idx = index // self.samples_per_mission
        remainder = index % self.samples_per_mission
        scale_idx = remainder // self.windows_per_scale
        record = self.records[mission_idx]
        window_length = int(CFG.window_lengths[scale_idx])

        data = self.loader.load(record["file_path"])
        T = int(len(data))
        if T >= window_length:
            start = int(rng.integers(0, T - window_length + 1))
        else:
            start = 0

        return self._build_sample(
            record=record,
            start=start,
            window_length=window_length,
            class_id=int(record["class_id"]),
            rng=rng,
        )


class TargetDomainDataset(Dataset):
    """
    目标域微调数据集。

    与源域数据集共享同一实现，仅替换输入 records。
    """

    def __init__(
        self,
        records: Sequence[Dict],
        zscore_mean: np.ndarray,
        zscore_std: np.ndarray,
        loader: MissionLoader,
        is_train: bool = True,
        base_seed: int = 42,
        enable_targeted_gan: bool = False,
        augment_mode: str = "full",
    ):
        self._inner = SourceDomainDataset(
            records=records,
            zscore_mean=zscore_mean,
            zscore_std=zscore_std,
            loader=loader,
            is_train=is_train,
            base_seed=base_seed,
            enable_targeted_gan=enable_targeted_gan,
            augment_mode=augment_mode,
        )

    def set_epoch(self, epoch: int) -> None:
        self._inner.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> Sample:
        return self._inner[index]


class DeterministicWindowDataset(Dataset):
    """
    验证/测试集数据集。

    规则：
    - 阶段窗口模式：直接复用 `SourceDomainDataset(is_train=False)` 的固定索引。
    - 传统模式：使用稳定 seed 生成固定窗口位置。
    """

    def __init__(
        self,
        records: Sequence[Dict],
        zscore_mean: np.ndarray,
        zscore_std: np.ndarray,
        loader: MissionLoader,
        windows_per_scale: int,
        base_seed: int = 42,
    ):
        self.use_stage_windowing = bool(getattr(CFG, "windows_enable_stage_windowing", False))
        if self.use_stage_windowing:
            self._stage_inner = SourceDomainDataset(
                records=records,
                zscore_mean=zscore_mean,
                zscore_std=zscore_std,
                loader=loader,
                is_train=False,
                base_seed=base_seed,
            )
            self._legacy_enabled = False
            return

        self._legacy_enabled = True
        self.records = list(records)
        self.mean = np.asarray(zscore_mean, dtype=np.float32).reshape(-1)
        self.std = np.maximum(np.asarray(zscore_std, dtype=np.float32).reshape(-1), 1e-8)
        self.loader = loader
        self.windows_per_scale = int(windows_per_scale)
        self.base_seed = int(base_seed)
        self.channel_names = tuple(ch.name for ch in CFG.channels)
        self._health_mask_cache: Dict[str, np.ndarray] = {}

        self.num_scales = len(CFG.window_lengths)
        self.samples_per_mission = self.windows_per_scale * self.num_scales
        self._total_len = len(self.records) * self.samples_per_mission

    def __len__(self) -> int:
        if self.use_stage_windowing:
            return len(self._stage_inner)
        return self._total_len

    def _stable_seed(self, mission_id: str, scale: int, k: int) -> int:
        # 验证/测试集固定采样位置需要跨进程稳定种子。
        return stable_seed_from_items(self.base_seed, mission_id, int(scale), int(k))

    def _slice_with_pad(
        self,
        array: np.ndarray,
        start: int,
        window_length: int,
        pad_value: float,
    ) -> Tuple[np.ndarray, int]:
        T = int(array.shape[0])
        C = int(array.shape[1])
        out = np.full((window_length, C), pad_value, dtype=np.float32)
        if start >= T:
            return out, 0
        end = min(start + window_length, T)
        real_len = max(end - start, 0)
        if real_len > 0:
            out[:real_len] = array[start:end]
        return out, int(real_len)

    def _get_health_mask_full(self, record: Dict, total_length: int) -> np.ndarray:
        file_path = record["file_path"]
        if file_path in self._health_mask_cache:
            return self._health_mask_cache[file_path]
        meta = self.loader.load_metadata(file_path)
        onset_idx = int(meta.fault_onset_idx)
        fault_code = int(record.get("fault_code", 10))
        if fault_code != 10 and onset_idx < 0:
            onset_idx = int(round(CFG.windows_normal_context_seconds * CFG.windows_sample_rate_hz))
        mask = generate_health_mask(
            total_length=total_length,
            fault_code=fault_code,
            fault_onset_idx=onset_idx,
            channel_names=self.channel_names,
        )
        self._health_mask_cache[file_path] = mask
        return mask

    def __getitem__(self, index: int) -> Sample:
        if self.use_stage_windowing:
            return self._stage_inner[index]

        mission_idx = index // self.samples_per_mission
        remainder = index % self.samples_per_mission
        scale_idx = remainder // self.windows_per_scale
        k = remainder % self.windows_per_scale

        record = self.records[mission_idx]
        window_length = int(CFG.window_lengths[scale_idx])
        seed = self._stable_seed(record["filename"], scale_idx, k)
        rng = np.random.default_rng(seed)

        data = self.loader.load(record["file_path"])
        T = int(len(data))
        if T >= window_length:
            start = int(rng.integers(0, T - window_length + 1))
        else:
            start = 0

        raw_window, real_len = self._slice_with_pad(data, start, window_length, pad_value=0.0)
        raw_window = _normalize_window_with_stats(
            raw_window=raw_window,
            mean=self.mean,
            std=self.std,
        )

        if bool(getattr(CFG, "use_cross_sensor_residuals", False)):
            raw_window = _append_cross_sensor_residual_features(
                normalized_window=raw_window,
                mean=self.mean,
                std=self.std,
            )

        if bool(getattr(CFG, "concat_health_mask_channels", True)):
            health_full = self._get_health_mask_full(record, total_length=T)
            health_window, _ = self._slice_with_pad(health_full, start, window_length, pad_value=1.0)
            x_np = np.concatenate([raw_window, health_window.astype(np.float32)], axis=1)
        else:
            x_np = raw_window

        x = torch.from_numpy(x_np.astype(np.float32))
        y = torch.tensor(int(record["class_id"]), dtype=torch.long)
        return Sample(x=x, length=int(real_len), y=y)


def collate_padded(
    batch: List[Sample],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    变长序列拼接函数。

    维度变换：
    - 输入：`batch` 中每个样本 `x_i` 形状 `[T_i, C]`
    - 输出：
      - `x_padded`: `[B, T_max, C]`
      - `lengths`: `[B]`
      - `y`: `[B]`
    """

    max_length = max(int(s.length) for s in batch)
    C = int(batch[0].x.shape[1])
    B = len(batch)

    x_padded = torch.zeros((B, max_length, C), dtype=torch.float32)
    lengths = torch.zeros((B,), dtype=torch.long)
    y = torch.zeros((B,), dtype=torch.long)

    for i, sample in enumerate(batch):
        L = int(sample.length)
        if L > 0:
            x_padded[i, :L, :] = sample.x[:L]
        lengths[i] = L
        y[i] = sample.y

    return x_padded, lengths, y
