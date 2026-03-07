"""
augment.py

物理感知数据增强框架。

设计目标:
1. 保持传感器数据的物理一致性（四元数归一化、轴间相关性）
2. 针对硬类（传感器故障 5,6,7）实施激进增强策略
3. 支持课程学习（随 epoch 渐进调整增强强度）
4. 自动监控与回退机制

增强策略（优先级）:
- P1: 时域增强（相关高斯抖动、偏置注入、线性漂移、增益缩放）
- P2: 频域增强（SpecAugment，TF2D 分支兼容）
- P3: 硬类激进策略（传感器故障模拟、同类 Mixup）
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple, Sequence

import numpy as np

from essence_forge.core.channel_layout import (
    LEGACY_ACCEL_INDICES,
    LEGACY_CORE_CHANNEL_NAMES,
    LEGACY_GYRO_INDICES,
    LEGACY_MAG_INDICES,
    LEGACY_POS_INDICES,
    LEGACY_QUATERNION_INDICES,
    LEGACY_VEL_INDICES,
    build_raw_sensor_groups,
)


# ============================================================================
# P1: 时域增强配置
# ============================================================================

@dataclass
class CorrelatedJitterConfig:
    """相关高斯抖动配置（保持轴间相关性）"""
    enable: bool = True
    prob: float = 0.7
    shared_sigma_ratio: float = 0.015  # 共享噪声比例 (1.5% sigma_ch)
    ind_sigma_ratio: float = 0.03       # 独立噪声比例 (3% sigma_ch)


@dataclass
class BiasInjectionConfig:
    """传感器偏置注入配置"""
    enable: bool = True
    prob: float = 0.4
    imu_bias_ratio: float = 0.05    # IMU 偏置比例 (5% sigma_ch)
    posvel_bias_ratio: float = 0.02  # pos/vel 偏置比例 (2% sigma_ch)


@dataclass
class LinearDriftConfig:
    """线性漂移注入配置"""
    enable: bool = True
    prob: float = 0.35
    imu_drift_range: Tuple[float, float] = (0.001, 0.01)  # 漂移斜率范围


@dataclass
class GainScalingConfig:
    """增益缩放配置"""
    enable: bool = True
    prob: float = 0.5
    imu_gain_range: Tuple[float, float] = (0.97, 1.03)
    posvel_gain_range: Tuple[float, float] = (0.98, 1.02)


@dataclass
class TimeDomainAugmentConfig:
    """时域增强总配置"""
    correlated_jitter: CorrelatedJitterConfig = field(default_factory=CorrelatedJitterConfig)
    bias_injection: BiasInjectionConfig = field(default_factory=BiasInjectionConfig)
    linear_drift: LinearDriftConfig = field(default_factory=LinearDriftConfig)
    gain_scaling: GainScalingConfig = field(default_factory=GainScalingConfig)


# ============================================================================
# P2: 频域增强配置
# ============================================================================

@dataclass
class TimeMaskConfig:
    """SpecAugment 时间掩码配置"""
    enable: bool = True
    prob: float = 0.35
    width_range: Tuple[int, int] = (2, 6)


@dataclass
class FreqMaskConfig:
    """SpecAugment 频率掩码配置"""
    enable: bool = True
    prob: float = 0.35
    width_range: Tuple[int, int] = (1, 2)


@dataclass
class SpectralTiltConfig:
    """频谱倾斜配置"""
    enable: bool = True
    prob: float = 0.2
    db_range: float = 1.5


@dataclass
class FreqDomainAugmentConfig:
    """频域增强总配置"""
    enable: bool = False  # 默认关闭，数据加载阶段启用
    time_mask: TimeMaskConfig = field(default_factory=TimeMaskConfig)
    freq_mask: FreqMaskConfig = field(default_factory=FreqMaskConfig)
    spectral_tilt: SpectralTiltConfig = field(default_factory=SpectralTiltConfig)


# ============================================================================
# P3: 硬类激进策略配置
# ============================================================================

@dataclass
class SensorFaultConfig:
    """传感器故障模拟配置"""
    bias_multiplier: float = 8.0          # 偏置扩大倍数（激进）
    stuck_segment_prob: float = 0.3       # 卡死段概率
    stuck_length_range: Tuple[int, int] = (30, 100)  # 卡死段长度范围


@dataclass
class HardClassMixupConfig:
    """硬类 Mixup 配置"""
    enable: bool = True
    prob: float = 0.35
    alpha: float = 0.3
    target_classes: Tuple[int, ...] = (5, 6, 7)
    same_fault_family_only: bool = False
    max_bank_size_per_class: int = 128


@dataclass
class ClassTargetedConfig:
    """
    类别定向增强配置。
    作用：
    1. 将增强重点转移到指定薄弱类别。
    2. 使用明确的概率上下界，避免增强强度失控。
    """

    enable: bool = True
    focus_classes: Tuple[int, ...] = (2, 4, 8, 9)
    min_prob: float = 0.45
    max_prob: float = 0.90


@dataclass
class ConfusionPairConfig:
    """
    混淆对定向增强配置。
    作用：
    1. 对已知混淆源类增加增强概率。
    2. 让增强资源集中到最容易混淆的决策边界。
    """

    enable: bool = True
    pairs: Tuple[Tuple[int, int], ...] = (
        (0, 10),
        (10, 0),
        (4, 10),
        (10, 4),
        (8, 10),
        (10, 8),
        (9, 10),
        (10, 9),
        (8, 9),
        (9, 8),
    )
    pair_boost: float = 0.30


@dataclass
class BoundaryJitterConfig:
    """
    故障注入边界增强配置。
    作用：
    1. 在注入前后边界窗口注入局部扰动。
    2. 强化模型对“正常->故障过渡区”模式的判别能力。
    """

    enable: bool = True
    near_onset_seconds: float = 1.5
    augment_prob: float = 0.60
    noise_sigma: float = 0.03


@dataclass
class FeatureBudgetConfig:
    """
    通道重要性预算配置。
    作用：
    1. 对高重要度通道采用更保守扰动。
    2. 对低重要度通道允许更大扰动以增强鲁棒性。
    """

    enable: bool = True
    protected_channels: Tuple[str, ...] = (
        "pos_x",
        "pos_y",
        "mag_x",
        "mag_y",
        "mag_z",
        "gyro_z",
        "accel_y",
        "accel_z",
    )
    protected_sigma_scale: float = 0.60
    unprotected_sigma_scale: float = 1.40


@dataclass
class SensorDropoutConfig:
    """
    传感器组遮蔽增强配置。
    作用：
    1. 训练期随机屏蔽一组原始传感器通道，降低单组过拟合。
    2. 通过保护组机制避免关键基础信号被无约束移除。
    """

    enable: bool = False
    prob: float = 0.15
    group_drop_prob: float = 0.30
    protected_groups: Tuple[str, ...] = ("accel",)


@dataclass
class HardClassConfig:
    """硬类激进策略总配置"""
    target_classes: Tuple[int, ...] = (5, 6, 7)  # 加速度计、陀螺仪、磁力计
    min_aug_prob: float = 0.8  # 硬类最小增强概率
    sensor_fault: SensorFaultConfig = field(default_factory=SensorFaultConfig)
    mixup: HardClassMixupConfig = field(default_factory=HardClassMixupConfig)


# ============================================================================
# 课程学习配置
# ============================================================================

@dataclass
class CurriculumConfig:
    """课程学习配置"""
    warmup_epochs: int = 8              # warmup 阶段 epoch 数
    full_aug_start_epoch: int = 8       # 完全增强起始 epoch
    decay_start_epoch_ratio: float = 0.85  # 衰减起始比例
    decay_factor: float = 0.5           # 衰减因子


@dataclass(frozen=True)
class AugContext:
    """
    增强上下文。
    字段定义与实现计划保持一致：
    - `fault_code`: 任务故障类型编码。
    - `fault_onset_idx`: 故障注入起点（全序列坐标）。
    - `window_start`: 当前窗口在全序列中的起点。
    - `window_length`: 当前窗口长度。
    - `channel_names`: 当前窗口通道名顺序，用于通道预算映射。
    """

    fault_code: int
    fault_onset_idx: int
    window_start: int
    window_length: int
    channel_names: Tuple[str, ...]


# ============================================================================
# 增强策略档位（用于回退机制）
# ============================================================================

@dataclass
class AugPolicy:
    """增强策略档位"""
    hard_prob: float           # 硬类增强概率
    bias_scale: float          # 偏置缩放因子
    stuck_min: int             # 卡死段最小长度
    stuck_max: int             # 卡死段最大长度
    name: str                  # 策略名称


# 四档策略定义
POLICIES: List[AugPolicy] = [
    AugPolicy(0.8, 8.0, 30, 100, "L0_aggressive"),
    AugPolicy(0.6, 4.0, 20, 70,  "L1_soft_rollback"),
    AugPolicy(0.4, 2.0, 15, 50,  "L2_safe"),
    AugPolicy(0.2, 1.0, 0,  0,   "L3_protect"),
]


# ============================================================================
# 风险监控指标
# ============================================================================

@dataclass
class RiskMetrics:
    """风险指标集合"""
    outlier_rate: float = 0.0         # 增强后 |z|>5 的比例
    corr_drift: float = 0.0           # 相关系数矩阵 Frobenius 漂移
    quat_norm_err: float = 0.0        # 四元数模长误差
    stuck_ratio: float = 0.0          # 硬类窗口近零导数点比例
    far: float = 0.0                  # 正常类误检率
    hcs: float = 0.0                  # 硬类召回率
    gen_gap: float = 0.0              # 训练-验证准确率差距
    hard_train_val_gap: float = 0.0   # 硬类训练-验证召回差距
    macro_f1_drop: float = 0.0        # 相对基线 macro-F1 下降
    ece_increase: float = 0.0         # 相对基线 ECE 增量
    grad_norm_p95_ratio: float = 1.0  # 梯度 95 分位/基线


# 预警/熔断阈值
WARN_THRESHOLDS = {
    "outlier_rate": 0.02,
    "corr_drift": 0.20,
    "quat_norm_err": 0.01,
    "stuck_ratio": 0.18,
    "far": 0.03,
    "hcs": 0.70,
    "gen_gap": 0.15,
    "hard_train_val_gap": 0.10,
    "macro_f1_drop": 0.015,
    "ece_increase": 0.02,
    "grad_norm_p95_ratio": 1.5,
}

CRIT_THRESHOLDS = {
    "outlier_rate": 0.05,
    "corr_drift": 0.30,
    "quat_norm_err": 0.02,
    "stuck_ratio": 0.25,
    "far": 0.05,
    "hcs": 0.60,
    "gen_gap": 0.25,
    "hard_train_val_gap": 0.15,
    "macro_f1_drop": 0.03,
    "ece_increase": 0.05,
    "grad_norm_p95_ratio": 2.0,
}


# ============================================================================
# 回退控制器
# ============================================================================

class AugRollbackController:
    """
    增强策略回退控制器。

    状态机设计:
    - 四档策略: L0(激进) -> L1(降级) -> L2(安全) -> L3(保护)
    - 防抖机制: 回退后冷却 3 个 epoch 不允许升档
    - 连续 3 个全绿 epoch 才允许升一档
    """

    def __init__(self, baseline_metrics: Optional[RiskMetrics] = None):
        self.level = 0  # 当前策略档位索引
        self.cooldown = 0  # 冷却计数
        self.warn_streak = 0  # 预警连续计数
        self.baseline = baseline_metrics or RiskMetrics()

    def get_policy(self) -> AugPolicy:
        """获取当前策略"""
        return POLICIES[self.level]

    def _judge(self, metrics: RiskMetrics) -> Tuple[int, int]:
        """
        判断当前指标是否触发预警/熔断。

        返回:
            (warn_count, crit_count)
        """
        warn_n = 0
        crit_n = 0

        for key, warn_thr in WARN_THRESHOLDS.items():
            val = getattr(metrics, key, 0.0)
            crit_thr = CRIT_THRESHOLDS.get(key, float('inf'))

            # 特殊处理: HCS 阈值是下限
            if key == "hcs":
                if val < crit_thr:
                    crit_n += 1
                elif val < warn_thr:
                    warn_n += 1
            else:
                if val > crit_thr:
                    crit_n += 1
                elif val > warn_thr:
                    warn_n += 1

        return warn_n, crit_n

    def step(self, metrics: RiskMetrics) -> AugPolicy:
        """
        根据当前指标更新策略档位。

        返回:
            更新后的策略
        """
        warn_n, crit_n = self._judge(metrics)

        # 冷却递减
        if self.cooldown > 0:
            self.cooldown -= 1

        # 熔断触发
        if crit_n > 0:
            self.level = min(self.level + 1, len(POLICIES) - 1)
            self.cooldown = 3
            self.warn_streak = 0
            print(f"[增强回退] 触发熔断(crit_n={crit_n})，降级到 {POLICIES[self.level].name}")
            return POLICIES[self.level]

        # 预警连续触发
        if warn_n >= 3:
            self.warn_streak += 1
            if self.warn_streak >= 2:
                self.level = min(self.level + 1, len(POLICIES) - 1)
                self.cooldown = 3
                self.warn_streak = 0
                print(f"[增强回退] 预警连续触发(warn_n={warn_n})，降级到 {POLICIES[self.level].name}")
                return POLICIES[self.level]
        else:
            self.warn_streak = 0

        # 尝试升档（仅在冷却结束且全绿时）
        if self.cooldown == 0 and warn_n == 0 and crit_n == 0:
            if self.level > 0:
                # 这里简化处理：不自动升档，保持当前档位
                # 可根据实际需求调整
                pass

        return POLICIES[self.level]

    def force_level(self, level: int) -> None:
        """强制设置档位（用于恢复最佳 checkpoint 后）"""
        self.level = max(0, min(level, len(POLICIES) - 1))
        self.cooldown = 3
        self.warn_streak = 0


# ============================================================================
# 物理感知增强器核心
# ============================================================================

class PhysicsAwareAugmentor:
    """
    物理一致性数据增强器。

    增强流程:
    1. 计算类别感知概率
    2. 应用时域增强（P1）
    3. 强制物理约束（四元数归一化）
    4. 返回增强后数据

    通道布局:
    - 对任意更宽输入，仅操作 legacy core 前 19 通道。
    - 新增的 actuator/motor/baro 通道在本次变更中不添加新的增强行为。
    """

    # 传感器组定义
    IMU_CHANNELS = LEGACY_ACCEL_INDICES + LEGACY_GYRO_INDICES
    MAG_CHANNELS = LEGACY_MAG_INDICES
    POSVEL_CHANNELS = LEGACY_POS_INDICES + LEGACY_VEL_INDICES
    QUAT_CHANNELS = LEGACY_QUATERNION_INDICES

    # 传感器组子分组（用于相关噪声）
    ACCEL_GROUP = LEGACY_ACCEL_INDICES
    GYRO_GROUP = LEGACY_GYRO_INDICES
    MAG_GROUP = LEGACY_MAG_INDICES
    SENSOR_GROUPS: Dict[str, Tuple[int, ...]] = build_raw_sensor_groups(
        LEGACY_CORE_CHANNEL_NAMES
    )

    def __init__(
        self,
        time_domain_cfg: Optional[TimeDomainAugmentConfig] = None,
        hard_class_cfg: Optional[HardClassConfig] = None,
        curriculum_cfg: Optional[CurriculumConfig] = None,
        class_targeted_cfg: Optional[ClassTargetedConfig] = None,
        confusion_pair_cfg: Optional[ConfusionPairConfig] = None,
        boundary_jitter_cfg: Optional[BoundaryJitterConfig] = None,
        feature_budget_cfg: Optional[FeatureBudgetConfig] = None,
        sensor_dropout_cfg: Optional[SensorDropoutConfig] = None,
        policy: Optional[AugPolicy] = None,
        base_sigma: float = 1.0,  # 标准化后数据的基础 sigma
    ):
        self.time_cfg = time_domain_cfg or TimeDomainAugmentConfig()
        self.hard_cfg = hard_class_cfg or HardClassConfig()
        self.curriculum_cfg = curriculum_cfg or CurriculumConfig()
        self.class_targeted_cfg = class_targeted_cfg or ClassTargetedConfig()
        self.confusion_pair_cfg = confusion_pair_cfg or ConfusionPairConfig()
        self.boundary_jitter_cfg = boundary_jitter_cfg or BoundaryJitterConfig()
        self.feature_budget_cfg = feature_budget_cfg or FeatureBudgetConfig()
        self.sensor_dropout_cfg = sensor_dropout_cfg or SensorDropoutConfig()
        self.policy = policy or POLICIES[0]
        self.base_sigma = base_sigma

        # 动态追踪各类召回率（用于类别感知概率）
        self.class_recall: Dict[int, float] = {}
        self._focus_class_set = {int(x) for x in self.class_targeted_cfg.focus_classes}
        self._confusion_sources = {int(src) for src, _ in self.confusion_pair_cfg.pairs}
        self._protected_channel_set = {str(name) for name in self.feature_budget_cfg.protected_channels}
        self._protected_group_set = {
            str(name).strip().lower() for name in self.sensor_dropout_cfg.protected_groups
        }

    def set_policy(self, policy: AugPolicy) -> None:
        """设置当前增强策略"""
        self.policy = policy

    def update_class_recall(self, class_recall: Dict[int, float]) -> None:
        """更新各类召回率（用于动态调整增强概率）"""
        self.class_recall = dict(class_recall)

    def _is_near_fault_boundary(self, context: Optional[AugContext]) -> bool:
        """
        判断当前窗口是否位于故障注入边界附近。
        为什么这样做：
        1. 仅在过渡区域提高增强强度，避免全局引入额外噪声。
        2. 与阶段切窗标签策略对齐，强化边界判别能力。
        """
        if context is None:
            return False
        if int(context.fault_code) == 10:
            return False
        onset_idx = int(context.fault_onset_idx)
        if onset_idx < 0:
            return False

        # 采样率固定 120Hz，与当前窗口配置保持一致。
        near_steps = int(round(float(self.boundary_jitter_cfg.near_onset_seconds) * 120.0))
        window_start = int(context.window_start)
        window_end = window_start + int(context.window_length)
        left = onset_idx - near_steps
        right = onset_idx + near_steps
        return not (window_end <= left or window_start >= right)

    def _get_class_aware_prob(
        self,
        y: int,
        epoch: int,
        context: Optional[AugContext] = None,
    ) -> float:
        """
        计算类别感知增强概率（扩展版）。
        组合策略：
        1. 原有课程学习与硬类动态策略；
        2. 类别定向范围约束；
        3. 混淆对来源类别概率提升；
        4. 边界窗口概率提升。
        """
        class_id = int(y)
        is_hard_class = class_id in self.hard_cfg.target_classes
        base_prob = 0.5

        if epoch < self.curriculum_cfg.warmup_epochs:
            warmup_factor = epoch / max(self.curriculum_cfg.warmup_epochs, 1)
            base_prob = 0.2 + 0.3 * warmup_factor
        elif epoch >= self.curriculum_cfg.decay_start_epoch_ratio * 100:
            decay_epochs = epoch - int(self.curriculum_cfg.decay_start_epoch_ratio * 100)
            decay_factor = self.curriculum_cfg.decay_factor ** (decay_epochs / 15)
            base_prob *= decay_factor

        if is_hard_class:
            min_prob = self.policy.hard_prob
            recall = self.class_recall.get(class_id, 0.5)
            lambda_factor = 0.3
            prob = float(np.clip(base_prob + lambda_factor * (1.0 - recall), min_prob, 0.95))
        else:
            prob = float(np.clip(base_prob, 0.2, 0.8))

        if self.class_targeted_cfg.enable and class_id in self._focus_class_set:
            prob = max(prob, float(self.class_targeted_cfg.min_prob))
            prob = min(prob, float(self.class_targeted_cfg.max_prob))

        if self.confusion_pair_cfg.enable and class_id in self._confusion_sources:
            prob += float(self.confusion_pair_cfg.pair_boost)

        if self.boundary_jitter_cfg.enable and self._is_near_fault_boundary(context):
            prob = max(prob, float(self.boundary_jitter_cfg.augment_prob))

        if self.class_targeted_cfg.enable and class_id in self._focus_class_set:
            prob = float(
                np.clip(
                    prob,
                    float(self.class_targeted_cfg.min_prob),
                    float(self.class_targeted_cfg.max_prob),
                )
            )
        return float(np.clip(prob, 0.0, 1.0))

    def augment(
        self,
        x: np.ndarray,
        y: int,
        epoch: int,
        rng: np.random.Generator,
        context: Optional[AugContext] = None,
    ) -> np.ndarray:
        """
        执行增强流程（重构版）。
        维度追踪：
        - 输入: [T, C]
        - 输出: [T, C]（shape 保持不变）
        """
        if x.ndim != 2:
            return x

        is_hard_class = int(y) in self.hard_cfg.target_classes
        p_aug = self._get_class_aware_prob(y, epoch, context=context)
        if rng.random() > p_aug:
            return x

        # 保存增强前基线，用于通道预算按 delta 缩放。
        x_ref = x.astype(np.float32, copy=False)
        x_aug = x.astype(np.float32, copy=True)

        if self.time_cfg.correlated_jitter.enable:
            x_aug = self._apply_correlated_jitter(x_aug, rng)
        if self.time_cfg.bias_injection.enable:
            x_aug = self._apply_bias_injection(x_aug, y, is_hard_class, rng)
        if self.time_cfg.linear_drift.enable:
            x_aug = self._apply_linear_drift(x_aug, rng)
        if self.time_cfg.gain_scaling.enable:
            x_aug = self._apply_gain_scaling(x_aug, rng)

        x_aug = self._apply_boundary_jitter(x_aug, rng=rng, context=context)
        if is_hard_class:
            x_aug = self._apply_sensor_fault(x_aug, y, rng)

        x_aug = self._apply_feature_budget(x_ref=x_ref, x_aug=x_aug, context=context)
        x_aug = self._apply_sensor_dropout(x=x_aug, rng=rng)
        x_aug = self._enforce_physics_constraints(x_aug)
        return x_aug

    def _apply_boundary_jitter(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        context: Optional[AugContext],
    ) -> np.ndarray:
        """
        在故障注入边界附近执行局部噪声增强。
        设计原因：
        1. 边界窗口是最容易产生类别混淆的区域。
        2. 仅对局部片段增强，尽量保持序列整体物理趋势。
        """
        cfg = self.boundary_jitter_cfg
        if not cfg.enable:
            return x
        if context is None:
            return x
        if not self._is_near_fault_boundary(context):
            return x
        if rng.random() > float(cfg.augment_prob):
            return x

        onset_local = int(context.fault_onset_idx) - int(context.window_start)
        if onset_local < 0 or onset_local >= int(context.window_length):
            return x

        radius = max(1, int(round(float(cfg.near_onset_seconds) * 120.0 * 0.5)))
        seg_start = max(0, onset_local - radius)
        seg_end = min(x.shape[0], onset_local + radius)
        if seg_start >= seg_end:
            return x

        # 仅扰动前 15 个原始时序通道，避免直接破坏四元数约束。
        num_channels = min(15, int(x.shape[1]))
        if num_channels <= 0:
            return x
        noise = rng.normal(
            0.0,
            float(cfg.noise_sigma),
            size=(seg_end - seg_start, num_channels),
        ).astype(np.float32)
        x[seg_start:seg_end, :num_channels] += noise
        return x

    def _apply_feature_budget(
        self,
        x_ref: np.ndarray,
        x_aug: np.ndarray,
        context: Optional[AugContext],
    ) -> np.ndarray:
        """
        按通道预算缩放增强扰动。
        实现方式：
        1. 先计算扰动增量 `delta = x_aug - x_ref`；
        2. 对保护通道和非保护通道分别乘以不同缩放系数；
        3. 回写 `x_ref + scaled_delta`，保持形状不变。
        """
        cfg = self.feature_budget_cfg
        if not cfg.enable:
            return x_aug
        if context is None:
            return x_aug
        if len(context.channel_names) == 0:
            return x_aug

        C = int(x_aug.shape[1])
        scales = np.full((C,), float(cfg.unprotected_sigma_scale), dtype=np.float32)
        for idx, name in enumerate(context.channel_names):
            if idx >= C:
                break
            if str(name) in self._protected_channel_set:
                scales[idx] = float(cfg.protected_sigma_scale)

        delta = x_aug - x_ref
        return (x_ref + delta * scales.reshape(1, -1)).astype(np.float32, copy=False)

    def _apply_sensor_dropout(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        传感器组随机遮蔽（仅作用于原始传感器通道）。
        设计约束：
        1. 保护组永不遮蔽。
        2. 触发后至少遮蔽 1 个非保护组，避免空触发。
        """
        cfg = self.sensor_dropout_cfg
        if not cfg.enable:
            return x
        if x.ndim != 2 or x.shape[1] <= 0:
            return x
        if rng.random() > float(cfg.prob):
            return x

        candidates = [
            name for name in self.SENSOR_GROUPS.keys()
            if str(name).strip().lower() not in self._protected_group_set
        ]
        if len(candidates) == 0:
            return x

        dropped: List[str] = []
        for group_name in candidates:
            if rng.random() < float(cfg.group_drop_prob):
                dropped.append(group_name)
        if len(dropped) == 0:
            dropped = [candidates[int(rng.integers(0, len(candidates)))]]

        channel_count = int(x.shape[1])
        for group_name in dropped:
            indices = [
                int(idx)
                for idx in self.SENSOR_GROUPS[group_name]
                if 0 <= int(idx) < channel_count
            ]
            if len(indices) > 0:
                x[:, indices] = 0.0
        return x

    def _apply_correlated_jitter(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        相关高斯抖动：保持轴间相关性。

        公式: eps = eps_shared + eps_ind
        - eps_shared: 同一传感器组共享
        - eps_ind: 每个通道独立
        """
        cfg = self.time_cfg.correlated_jitter
        if rng.random() > cfg.prob:
            return x

        T, C = x.shape
        sigma = self.base_sigma

        # 生成共享噪声
        shared_sigma = cfg.shared_sigma_ratio * sigma

        # Accel 组共享噪声
        accel_shared = rng.normal(0.0, shared_sigma, (T, 1))
        x[:, 0:3] += accel_shared

        # Gyro 组共享噪声
        gyro_shared = rng.normal(0.0, shared_sigma, (T, 1))
        x[:, 3:6] += gyro_shared

        # Mag 组共享噪声
        mag_shared = rng.normal(0.0, shared_sigma, (T, 1))
        x[:, 6:9] += mag_shared

        # 独立噪声（仅 IMU 通道）
        ind_sigma = cfg.ind_sigma_ratio * sigma
        ind_noise = rng.normal(0.0, ind_sigma, (T, 9))
        x[:, 0:9] += ind_noise

        return x

    def _apply_bias_injection(
        self,
        x: np.ndarray,
        y: int,
        is_hard_class: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        传感器偏置注入：模拟校准偏差。

        硬类使用扩大的偏置范围（bias_multiplier）。
        """
        cfg = self.time_cfg.bias_injection
        if rng.random() > cfg.prob:
            return x

        sigma = self.base_sigma

        # IMU 偏置
        imu_bias_ratio = cfg.imu_bias_ratio
        if is_hard_class:
            imu_bias_ratio *= self.policy.bias_scale

        imu_bias = rng.uniform(
            -imu_bias_ratio * sigma,
            imu_bias_ratio * sigma,
            (6,),
        )
        x[:, 0:6] += imu_bias

        # pos/vel 偏置
        posvel_bias_ratio = cfg.posvel_bias_ratio
        posvel_bias = rng.uniform(
            -posvel_bias_ratio * sigma,
            posvel_bias_ratio * sigma,
            (6,),
        )
        x[:, 9:15] += posvel_bias

        return x

    def _apply_linear_drift(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        线性漂移注入：模拟热漂移/积分漂移。

        仅作用于 IMU 通道。
        """
        cfg = self.time_cfg.linear_drift
        if rng.random() > cfg.prob:
            return x

        T = x.shape[0]
        sigma = self.base_sigma

        # 生成漂移斜率
        drift_min, drift_max = cfg.imu_drift_range
        slopes = rng.uniform(drift_min, drift_max, (6,)) * sigma
        signs = rng.choice([-1.0, 1.0], (6,))
        slopes = slopes * signs

        # 线性累积
        t_normalized = np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(-1, 1)
        drift = (slopes * t_normalized).astype(np.float32)
        x[:, 0:6] += drift

        return x

    def _apply_gain_scaling(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        增益缩放：模拟传感器标定误差。

        组级别共享增益（同一传感器组）。
        """
        cfg = self.time_cfg.gain_scaling
        if rng.random() > cfg.prob:
            return x

        # IMU 组增益（accel + gyro 共享）
        imu_gain = rng.uniform(*cfg.imu_gain_range)
        x[:, 0:6] *= imu_gain

        # Mag 组增益
        mag_gain = rng.uniform(*cfg.imu_gain_range)
        x[:, 6:9] *= mag_gain

        # pos/vel 组增益
        posvel_gain = rng.uniform(*cfg.posvel_gain_range)
        x[:, 9:15] *= posvel_gain

        return x

    def _apply_sensor_fault(
        self,
        x: np.ndarray,
        y: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        传感器故障模拟（仅硬类）。

        类别映射:
        - 5 (accel): 偏置/卡死
        - 6 (gyro): 漂移/削波
        - 7 (mag): 硬铁/软铁效应
        """
        cfg = self.hard_cfg.sensor_fault

        # 卡死段注入（概率触发）
        if rng.random() < cfg.stuck_segment_prob:
            stuck_min = max(self.policy.stuck_min, 1)
            stuck_max = max(self.policy.stuck_max, stuck_min)

            T = x.shape[0]
            stuck_len = int(rng.integers(stuck_min, stuck_max + 1))
            start_idx = int(rng.integers(0, max(T - stuck_len, 1)))

            if int(y) == 5:  # Accel 卡死
                x[start_idx:start_idx + stuck_len, 0:3] = x[start_idx, 0:3]
            elif int(y) == 6:  # Gyro 卡死
                x[start_idx:start_idx + stuck_len, 3:6] = x[start_idx, 3:6]
            elif int(y) == 7:  # Mag 卡死
                x[start_idx:start_idx + stuck_len, 6:9] = x[start_idx, 6:9]

        # 传感器特定故障
        if int(y) == 6:  # Gyro: 随机游走漂移
            self._apply_gyro_random_walk(x, rng)
        elif int(y) == 7:  # Mag: 软铁效应
            self._apply_mag_soft_iron(x, rng)

        return x

    def _apply_gyro_random_walk(self, x: np.ndarray, rng: np.random.Generator) -> None:
        """陀螺仪随机游走漂移"""
        T = x.shape[0]
        sigma = self.base_sigma

        # 随机游走幅值
        walk_amp = rng.uniform(0.6, 2.4) * sigma * self.policy.bias_scale

        # 累积随机游走
        steps = rng.normal(0.0, walk_amp / np.sqrt(T), (T, 3))
        walk = np.cumsum(steps, axis=0).astype(np.float32)
        x[:, 3:6] += walk

        # 削波（模拟传感器饱和）
        clip_thr = rng.uniform(3.0, 4.0) * sigma
        x[:, 3:6] = np.clip(x[:, 3:6], -clip_thr, clip_thr)

    def _apply_mag_soft_iron(self, x: np.ndarray, rng: np.random.Generator) -> None:
        """磁力计软铁效应（矩阵扰动）"""
        sigma = self.base_sigma

        # 软铁矩阵（小扰动）
        scale = 0.4 * sigma * self.policy.bias_scale
        soft_iron = np.eye(3) + rng.uniform(-scale, scale, (3, 3))
        x[:, 6:9] = x[:, 6:9] @ soft_iron.T

    def _enforce_physics_constraints(self, x: np.ndarray) -> np.ndarray:
        """
        强制物理约束：
        1. 四元数归一化
        2. 符号连续性

        四元数通道: legacy core 中的 q0/q1/q2/q3
        """
        if x.shape[1] < 19:
            return x

        q = x[:, list(self.QUAT_CHANNELS)]

        # 归一化
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)  # 防止除零
        q_normalized = q / norm

        # 符号连续性（确保相邻帧四元数符号一致）
        for t in range(1, len(q)):
            if np.dot(q_normalized[t], q_normalized[t - 1]) < 0:
                q_normalized[t] *= -1

        x[:, list(self.QUAT_CHANNELS)] = q_normalized
        return x


# ============================================================================
# P2: 频域增强（数据加载阶段）
# ============================================================================

class FreqDomainAugmentor:
    """
    频域增强器（用于 TF2D 分支）。

    在数据加载阶段对 STFT 输出应用增强。
    """

    def __init__(self, cfg: Optional[FreqDomainAugmentConfig] = None):
        self.cfg = cfg or FreqDomainAugmentConfig()

    def augment(
        self,
        spec: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        对频谱应用增强。

        参数:
            spec: STFT 输出 [F, T] 或 [C, F, T]
            rng: 随机数生成器

        返回:
            增强后的频谱
        """
        if not self.cfg.enable:
            return spec

        spec_aug = spec.astype(np.float32, copy=True)

        # 处理多通道情况
        if spec_aug.ndim == 3:
            for c in range(spec_aug.shape[0]):
                spec_aug[c] = self._augment_single(spec_aug[c], rng)
        else:
            spec_aug = self._augment_single(spec_aug, rng)

        return spec_aug

    def _augment_single(self, spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """对单个频谱应用增强"""
        F, T = spec.shape

        # 时间掩码
        if self.cfg.time_mask.enable and rng.random() < self.cfg.time_mask.prob:
            width = int(rng.integers(*self.cfg.time_mask.width_range))
            t_start = int(rng.integers(0, max(T - width, 1)))
            spec[:, t_start:t_start + width] = 0.0

        # 频率掩码
        if self.cfg.freq_mask.enable and rng.random() < self.cfg.freq_mask.prob:
            width = int(rng.integers(*self.cfg.freq_mask.width_range))
            f_start = int(rng.integers(0, max(F - width, 1)))
            spec[f_start:f_start + width, :] = 0.0

        # 频谱倾斜
        if self.cfg.spectral_tilt.enable and rng.random() < self.cfg.spectral_tilt.prob:
            tilt_db = rng.uniform(-self.cfg.spectral_tilt.db_range, self.cfg.spectral_tilt.db_range)
            tilt_linear = 10.0 ** (tilt_db / 20.0)
            freq_weights = np.linspace(1.0 / tilt_linear, tilt_linear, F).reshape(-1, 1)
            spec *= freq_weights

        return spec


# ============================================================================
# P3: 硬类 Mixup（数据集级别）
# ============================================================================

class HardClassMixupAugmentor:
    """
    硬类 Mixup 增强器。
    关键点：
    1. 支持可配置目标类别，不再写死 5/6/7。
    2. 可选同故障家族约束，降低跨物理机理混合带来的标签噪声。
    """

    def __init__(self, cfg: Optional[HardClassMixupConfig] = None):
        self.cfg = cfg or HardClassMixupConfig()
        self.max_bank_size_per_class = max(1, int(self.cfg.max_bank_size_per_class))
        self._sample_bank: Dict[int, Deque[np.ndarray]] = {}
        self._target_class_set = {int(x) for x in self.cfg.target_classes}

    def _fault_family(self, class_id: int) -> str:
        """
        计算故障家族标签。
        为什么这样做：
        1. 支持“同故障家族”约束下的 mixup 采样。
        2. 保持类别映射规则简单且可解释。
        """
        cid = int(class_id)
        if cid in (2, 4):
            return "power"
        if cid in (8, 9):
            return "env_nav"
        if cid in (5, 6, 7):
            return "sensor"
        if cid == 10:
            return "normal"
        return f"class_{cid}"

    def register_sample(self, y: int, x: np.ndarray) -> None:
        """注册样本到样本库。"""
        class_id = int(y)
        if class_id not in self._target_class_set:
            return
        if class_id not in self._sample_bank:
            self._sample_bank[class_id] = deque(maxlen=self.max_bank_size_per_class)
        self._sample_bank[class_id].append(x.astype(np.float32, copy=False))

    def maybe_mixup(
        self,
        x: np.ndarray,
        y: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        按配置执行 Mixup（概率触发）。
        返回：
            混合后的样本（或原样本）。
        """
        if not self.cfg.enable:
            return x

        class_id = int(y)
        if class_id not in self._target_class_set:
            return x
        if rng.random() > self.cfg.prob:
            return x

        candidate_classes: List[int] = []
        if self.cfg.same_fault_family_only:
            family = self._fault_family(class_id)
            for cid in self._sample_bank.keys():
                if self._fault_family(cid) == family:
                    candidate_classes.append(int(cid))
        else:
            candidate_classes = [int(cid) for cid in self._sample_bank.keys()]

        candidate_classes = [
            cid for cid in candidate_classes if len(self._sample_bank.get(int(cid), ())) > 0
        ]
        if len(candidate_classes) == 0:
            return x

        class_sizes = np.asarray(
            [len(self._sample_bank[int(cid)]) for cid in candidate_classes],
            dtype=np.float64,
        )
        size_sum = float(class_sizes.sum())
        if size_sum <= 0.0:
            return x
        class_probs = class_sizes / size_sum
        selected_class = int(
            candidate_classes[int(rng.choice(len(candidate_classes), p=class_probs))]
        )
        selected_bank = self._sample_bank[selected_class]
        other = selected_bank[int(rng.integers(0, len(selected_bank)))]
        if other.shape != x.shape:
            return x

        alpha = float(self.cfg.alpha)
        lam = float(rng.beta(alpha, alpha))
        mixed = lam * x + (1.0 - lam) * other
        return mixed.astype(np.float32, copy=False)

    def clear_bank(self) -> None:
        """清空样本库（每个 epoch 开始时调用）。"""
        self._sample_bank.clear()


def build_augmentor_from_config(
    config: Any,
    policy: Optional[AugPolicy] = None,
) -> PhysicsAwareAugmentor:
    """
    从配置对象构建物理感知增强器。

    参数:
        config: 配置对象（需包含增强相关字段）
        policy: 可选的策略档位

    返回:
        PhysicsAwareAugmentor 实例
    """
    # 解析时域配置
    time_cfg = TimeDomainAugmentConfig(
        correlated_jitter=CorrelatedJitterConfig(
            enable=bool(getattr(config, "augment_correlated_jitter_enable", True)),
            prob=float(getattr(config, "augment_correlated_jitter_prob", 0.7)),
            shared_sigma_ratio=float(getattr(config, "augment_correlated_jitter_shared_sigma", 0.015)),
            ind_sigma_ratio=float(getattr(config, "augment_correlated_jitter_ind_sigma", 0.03)),
        ),
        bias_injection=BiasInjectionConfig(
            enable=bool(getattr(config, "augment_bias_injection_enable", True)),
            prob=float(getattr(config, "augment_bias_injection_prob", 0.4)),
            imu_bias_ratio=float(getattr(config, "augment_bias_injection_imu_ratio", 0.05)),
            posvel_bias_ratio=float(getattr(config, "augment_bias_injection_posvel_ratio", 0.02)),
        ),
        linear_drift=LinearDriftConfig(
            enable=bool(getattr(config, "augment_linear_drift_enable", True)),
            prob=float(getattr(config, "augment_linear_drift_prob", 0.35)),
            imu_drift_range=tuple(getattr(config, "augment_linear_drift_range", (0.001, 0.01))),
        ),
        gain_scaling=GainScalingConfig(
            enable=bool(getattr(config, "augment_gain_scaling_enable", True)),
            prob=float(getattr(config, "augment_gain_scaling_prob", 0.5)),
            imu_gain_range=tuple(getattr(config, "augment_gain_scaling_imu_range", (0.97, 1.03))),
            posvel_gain_range=tuple(getattr(config, "augment_gain_scaling_posvel_range", (0.98, 1.02))),
        ),
    )

    # 解析硬类配置
    hard_cfg = HardClassConfig(
        target_classes=tuple(getattr(config, "augment_hard_class_targets", (5, 6, 7))),
        min_aug_prob=float(getattr(config, "augment_hard_class_min_prob", 0.8)),
        sensor_fault=SensorFaultConfig(
            bias_multiplier=float(getattr(config, "augment_sensor_fault_bias_mult", 8.0)),
            stuck_segment_prob=float(getattr(config, "augment_sensor_fault_stuck_prob", 0.3)),
            stuck_length_range=tuple(getattr(config, "augment_sensor_fault_stuck_range", (30, 100))),
        ),
        mixup=HardClassMixupConfig(
            enable=bool(
                getattr(
                    config,
                    "augment_class_mixup_enable",
                    getattr(config, "augment_hard_class_mixup_enable", True),
                )
            ),
            prob=float(
                getattr(
                    config,
                    "augment_class_mixup_prob",
                    getattr(config, "augment_hard_class_mixup_prob", 0.35),
                )
            ),
            alpha=float(
                getattr(
                    config,
                    "augment_class_mixup_alpha",
                    getattr(config, "augment_hard_class_mixup_alpha", 0.3),
                )
            ),
            target_classes=tuple(
                getattr(
                    config,
                    "augment_class_mixup_target_classes",
                    getattr(config, "augment_hard_class_targets", (5, 6, 7)),
                )
            ),
            same_fault_family_only=bool(
                getattr(config, "augment_class_mixup_same_fault_family_only", False)
            ),
            max_bank_size_per_class=int(
                getattr(config, "augment_class_mixup_max_bank_size_per_class", 128)
            ),
        ),
    )

    # 解析课程学习配置
    curriculum_cfg = CurriculumConfig(
        warmup_epochs=int(getattr(config, "augment_curriculum_warmup", 8)),
        full_aug_start_epoch=int(getattr(config, "augment_curriculum_full_start", 8)),
        decay_start_epoch_ratio=float(getattr(config, "augment_curriculum_decay_ratio", 0.85)),
        decay_factor=float(getattr(config, "augment_curriculum_decay_factor", 0.5)),
    )

    class_targeted_cfg = ClassTargetedConfig(
        enable=bool(getattr(config, "augment_class_targeted_enable", True)),
        focus_classes=tuple(getattr(config, "augment_class_targeted_focus_classes", (2, 4, 8, 9))),
        min_prob=float(getattr(config, "augment_class_targeted_min_prob", 0.45)),
        max_prob=float(getattr(config, "augment_class_targeted_max_prob", 0.90)),
    )

    confusion_pair_cfg = ConfusionPairConfig(
        enable=bool(getattr(config, "augment_confusion_pair_enable", True)),
        pairs=tuple(
            tuple(int(v) for v in pair)
            for pair in getattr(
                config,
                "augment_confusion_pair_pairs",
                (
                    (0, 10),
                    (10, 0),
                    (4, 10),
                    (10, 4),
                    (8, 10),
                    (10, 8),
                    (9, 10),
                    (10, 9),
                    (8, 9),
                    (9, 8),
                ),
            )
        ),
        pair_boost=float(getattr(config, "augment_confusion_pair_pair_boost", 0.30)),
    )

    boundary_jitter_cfg = BoundaryJitterConfig(
        enable=bool(getattr(config, "augment_boundary_jitter_enable", True)),
        near_onset_seconds=float(getattr(config, "augment_boundary_jitter_near_onset_seconds", 1.5)),
        augment_prob=float(getattr(config, "augment_boundary_jitter_augment_prob", 0.60)),
        noise_sigma=float(getattr(config, "augment_boundary_jitter_noise_sigma", 0.03)),
    )

    feature_budget_cfg = FeatureBudgetConfig(
        enable=bool(getattr(config, "augment_feature_budget_enable", True)),
        protected_channels=tuple(
            str(v)
            for v in getattr(
                config,
                "augment_feature_budget_protected_channels",
                ("pos_x", "pos_y", "mag_x", "mag_y", "mag_z", "gyro_z", "accel_y", "accel_z"),
            )
        ),
        protected_sigma_scale=float(getattr(config, "augment_feature_budget_protected_sigma_scale", 0.60)),
        unprotected_sigma_scale=float(getattr(config, "augment_feature_budget_unprotected_sigma_scale", 1.40)),
    )
    sensor_dropout_cfg = SensorDropoutConfig(
        enable=bool(getattr(config, "augment_sensor_dropout_enable", False)),
        prob=float(getattr(config, "augment_sensor_dropout_prob", 0.15)),
        group_drop_prob=float(getattr(config, "augment_sensor_dropout_group_drop_prob", 0.30)),
        protected_groups=tuple(
            str(v)
            for v in getattr(
                config,
                "augment_sensor_dropout_protected_groups",
                ("accel",),
            )
        ),
    )

    return PhysicsAwareAugmentor(
        time_domain_cfg=time_cfg,
        hard_class_cfg=hard_cfg,
        curriculum_cfg=curriculum_cfg,
        class_targeted_cfg=class_targeted_cfg,
        confusion_pair_cfg=confusion_pair_cfg,
        boundary_jitter_cfg=boundary_jitter_cfg,
        feature_budget_cfg=feature_budget_cfg,
        sensor_dropout_cfg=sensor_dropout_cfg,
        policy=policy,
    )


# ============================================================================
# 工具函数
# ============================================================================

def validate_physics_constraints(x: np.ndarray) -> Tuple[bool, float]:
    """
    验证物理约束。

    返回:
        (is_valid, quat_norm_error)
    """
    if x.ndim != 2 or x.shape[1] < 19:
        return True, 0.0

    q = x[:, list(LEGACY_QUATERNION_INDICES)]
    norms = np.linalg.norm(q, axis=1)
    quat_norm_err = float(np.mean(np.abs(norms - 1.0)))

    is_valid = quat_norm_err < 0.02
    return is_valid, quat_norm_err


def compute_augmentation_metrics(
    original: np.ndarray,
    augmented: np.ndarray,
) -> Dict[str, float]:
    """
    计算增强前后数据分布的指标。

    返回:
        指标字典
    """
    metrics = {}

    # Outlier rate (|z| > 5)
    z_scores = (augmented - original.mean(axis=0)) / (original.std(axis=0) + 1e-8)
    metrics["outlier_rate"] = float(np.mean(np.abs(z_scores) > 5))

    # Correlation drift
    if original.shape[0] > 1:
        corr_orig = np.corrcoef(original.T)
        corr_aug = np.corrcoef(augmented.T)
        if corr_orig.ndim == 2 and corr_aug.ndim == 2:
            metrics["corr_drift"] = float(np.linalg.norm(corr_aug - corr_orig, 'fro'))
        else:
            metrics["corr_drift"] = 0.0
    else:
        metrics["corr_drift"] = 0.0

    # Quaternary norm error
    _, metrics["quat_norm_err"] = validate_physics_constraints(augmented)

    return metrics

