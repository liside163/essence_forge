"""
rflymad_io.py

RflyMAD Mission 数据加载模块。
核心职责：
1. 读取并返回模型输入特征矩阵 `data`，形状 `[T, C]`；
2. 读取故障状态序列、故障起点和采样率元信息；
3. 复用离线缓存（`.npy/.npz`），减少训练阶段重复 CSV 解析；
4. 提供 Z-score 统计量计算接口。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import sys


from essence_forge.core.runtime_config import CFG


@dataclass(frozen=True)
class MissionMetadata:
    """
    Mission 元信息。
    维度说明：
    - `fault_state`: `[T]`，每个时刻的故障状态编码；
    - `fault_onset_idx`: 故障首次出现的时间索引，若不存在故障则为 -1；
    - `sample_rate_hz`: 估计采样率（Hz）。
    """

    fault_state: np.ndarray
    fault_onset_idx: int
    sample_rate_hz: int


class MissionLoader:
    """
    Mission 加载器（带 LRU 缓存）。
    设计要点：
    - 优先读取离线缓存 `cache_dir/Case_xxx.npy(npz)`，避免反复 CSV 解析；
    - 若缓存不存在或形状不匹配，自动回退 CSV；
    - 元信息仍从 CSV 读取（尽量只读必需列），保证阶段切窗标签逻辑正确。
    """

    def __init__(self, max_cache_items: int = 32):
        self._max_cache = int(max_cache_items)
        self._cache: Dict[str, Dict[str, object]] = {}
        self._cache_order: List[str] = []
        self._warned_keys: Set[str] = set()

    def load(self, file_path: str) -> np.ndarray:
        """
        读取 mission 输入特征。
        返回维度：
        - `data`: `[T, C]`，其中 `C=CFG.input_dim`（当前工程默认 19）。
        """

        pack = self._load_pack(file_path)
        return pack["data"]  # type: ignore[return-value]

    def load_metadata(self, file_path: str) -> MissionMetadata:
        """
        读取 mission 元信息（fault_state + onset + sample_rate）。
        """

        pack = self._load_pack(file_path)
        return MissionMetadata(
            fault_state=pack["fault_state"],  # type: ignore[arg-type]
            fault_onset_idx=int(pack["fault_onset_idx"]),  # type: ignore[arg-type]
            sample_rate_hz=int(pack["sample_rate_hz"]),  # type: ignore[arg-type]
        )

    def load_fault_state(self, file_path: str) -> np.ndarray:
        """仅返回故障状态序列，维度 `[T]`。"""

        return self.load_metadata(file_path).fault_state

    def load_sample_rate_hz(self, file_path: str) -> int:
        """仅返回采样率（Hz）。"""

        return self.load_metadata(file_path).sample_rate_hz

    def clear_cache(self) -> None:
        """清空内存 LRU 缓存。"""

        self._cache.clear()
        self._cache_order.clear()

    def _load_pack(self, file_path: str) -> Dict[str, object]:
        """
        加载并缓存单个 mission 的完整解析结果。
        缓存结构：
        - `data`: `[T, C]`
        - `fault_state`: `[T]`
        - `fault_onset_idx`: `int`
        - `sample_rate_hz`: `int`
        """

        normalized_path = self._normalize_file_path(file_path)
        if normalized_path in self._cache:
            # LRU 命中后移动到队尾。
            self._cache_order.remove(normalized_path)
            self._cache_order.append(normalized_path)
            return self._cache[normalized_path]

        csv_path = Path(normalized_path)

        # -----------------------------
        # 1) 优先尝试离线缓存（核心提速点）
        # -----------------------------
        cached_data = self._load_data_from_offline_cache(csv_path)
        if cached_data is not None:
            # 仅补充元信息所需列，避免再次读取全量传感器列。
            fault_state, sample_rate_hz = self._load_metadata_from_csv(
                csv_path=csv_path,
                expected_length=int(cached_data.shape[0]),
            )
            fault_onset_idx = self._compute_fault_onset_idx(fault_state)
            pack: Dict[str, object] = {
                "data": cached_data,
                "fault_state": fault_state,
                "fault_onset_idx": fault_onset_idx,
                "sample_rate_hz": sample_rate_hz,
            }
            self._put_cache(normalized_path, pack)
            return pack

        # -----------------------------
        # 2) 缓存不可用时，回退 CSV 解析
        # -----------------------------
        df = pd.read_csv(csv_path)
        data = self._extract_channels(df)  # [T, C]
        fault_state = self._extract_fault_state(df)  # [T]
        fault_state = self._align_vector_length(
            values=fault_state,
            expected_length=int(data.shape[0]),
            fill_value=0,
        )
        fault_onset_idx = self._compute_fault_onset_idx(fault_state)
        sample_rate_hz = self._estimate_sample_rate_hz(df)

        pack = {
            "data": data,
            "fault_state": fault_state,
            "fault_onset_idx": fault_onset_idx,
            "sample_rate_hz": sample_rate_hz,
        }
        self._put_cache(normalized_path, pack)
        return pack

    def _put_cache(self, normalized_path: str, pack: Dict[str, object]) -> None:
        """
        写入 LRU 缓存并执行淘汰。
        """

        self._cache[normalized_path] = pack
        self._cache_order.append(normalized_path)
        while len(self._cache_order) > self._max_cache:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

    @staticmethod
    def _normalize_file_path(file_path: str) -> str:
        """
        兼容 WSL/Windows 路径差异，归一化到当前运行环境可读路径。
        """

        raw = str(file_path).strip()
        if len(raw) >= 3 and raw[1] == ":" and raw[2] in ("/", "\\"):
            if os.name != "nt":
                drive = raw[0].lower()
                tail = raw[3:].replace("\\", "/")
                return str(Path(f"/mnt/{drive}/{tail}").resolve())
            return str(Path(raw).resolve())

        path_obj = Path(raw).expanduser()
        if path_obj.is_absolute():
            return str(path_obj.resolve())
        return str((CFG.project_dir / path_obj).resolve())

    def _load_data_from_offline_cache(self, csv_path: Path) -> Optional[np.ndarray]:
        """
        读取离线缓存数组。
        成功返回：
        - `x`: `[T, C]`，且 `C` 必须等于 `CFG.input_dim`；
        失败返回 `None`（由上层回退到 CSV）。
        """

        if not bool(getattr(CFG, "cache_enabled", False)):
            return None

        cache_dir_obj = getattr(CFG, "cache_dir", None)
        if cache_dir_obj is None:
            return None
        cache_dir = Path(str(cache_dir_obj)).expanduser().resolve()

        cache_format = str(getattr(CFG, "cache_format", "npy")).lower().strip()
        if cache_format not in {"npy", "npz"}:
            self._warn_once(
                key=f"cache.format.{cache_format}",
                message=f"[WARN] cache.format={cache_format} 非法，已回退 CSV。",
            )
            return None

        suffix = ".npy" if cache_format == "npy" else ".npz"
        cache_path = (cache_dir / f"{csv_path.stem}{suffix}").resolve()
        if not cache_path.exists():
            return None

        try:
            if cache_format == "npy":
                mmap_mode = "r" if bool(getattr(CFG, "cache_mmap", False)) else None
                x = np.load(cache_path, mmap_mode=mmap_mode)
            else:
                with np.load(cache_path, allow_pickle=False) as payload:
                    if "x" not in payload:
                        self._warn_once(
                            key=f"cache.no_x.{cache_path}",
                            message=f"[WARN] 缓存文件缺少键 `x`：{cache_path}，已回退 CSV。",
                        )
                        return None
                    x = payload["x"]
            x = np.asarray(x, dtype=np.float32)
        except Exception as exc:
            self._warn_once(
                key=f"cache.load_fail.{cache_path}",
                message=f"[WARN] 读取缓存失败，已回退 CSV：{cache_path}，错误：{exc!r}",
            )
            return None

        if x.ndim != 2:
            self._warn_once(
                key=f"cache.ndim.{cache_path}",
                message=f"[WARN] 缓存维度异常（期望 2D，实际 {x.ndim}D）：{cache_path}，已回退 CSV。",
            )
            return None

        expected_c = int(getattr(CFG, "input_dim", x.shape[1]))
        if int(x.shape[1]) == expected_c:
            return x

        # 兼容场景：`UAV_muilt-scale/cache_npy` 的 20 通道缓存适配到当前 19 通道输入。
        adapted_x = self._adapt_cache_20_to_target_19(x=x, expected_c=expected_c)
        if adapted_x is not None:
            return adapted_x

        if int(x.shape[1]) != expected_c:
            self._warn_once(
                key=f"cache.shape.{cache_path}",
                message=(
                    f"[WARN] 缓存通道数不匹配（缓存 C={x.shape[1]}，配置 C={expected_c}）："
                    f"{cache_path}，已回退 CSV。"
                ),
            )
            return None

        return x

    def _adapt_cache_20_to_target_19(self, x: np.ndarray, expected_c: int) -> Optional[np.ndarray]:
        """
        适配多尺度工程 20 通道缓存到当前工程 19 通道布局。
        适配规则：
        - 可直接映射的通道：acc/gyro/mag/q/pos；
        - 缺失的速度通道：由位置通道按时间差分近似。
        若配置通道名称与预期不一致，则返回 `None` 交给上层回退 CSV。
        """

        if expected_c != 19 or int(x.shape[1]) != 20:
            return None

        source_name_to_idx = {
            "gyro_x": 0,
            "gyro_y": 1,
            "gyro_z": 2,
            "acc_x": 3,
            "acc_y": 4,
            "acc_z": 5,
            "mag_x": 6,
            "mag_y": 7,
            "mag_z": 8,
            "q0": 9,
            "q1": 10,
            "q2": 11,
            "q3": 12,
            "rpm1": 13,
            "rpm2": 14,
            "rpm3": 15,
            "rpm4": 16,
            "pos_x": 17,
            "pos_y": 18,
            "pos_z": 19,
        }
        target_alias = {
            "accel_x": "acc_x",
            "accel_y": "acc_y",
            "accel_z": "acc_z",
        }

        target_names = [str(ch.name) for ch in getattr(CFG, "channels", ())]
        if len(target_names) != expected_c:
            return None

        t_len = int(x.shape[0])
        out = np.zeros((t_len, expected_c), dtype=np.float32)

        sample_rate_hz = max(int(getattr(CFG, "windows_sample_rate_hz", 120)), 1)
        dt = 1.0 / float(sample_rate_hz)

        for target_idx, target_name in enumerate(target_names):
            if target_name in {"vel_x", "vel_y", "vel_z"}:
                pos_name = target_name.replace("vel_", "pos_")
                pos_idx = source_name_to_idx.get(pos_name)
                if pos_idx is None:
                    return None
                out[:, target_idx] = np.gradient(x[:, pos_idx].astype(np.float32), dt).astype(np.float32)
                continue

            source_name = target_alias.get(target_name, target_name)
            source_idx = source_name_to_idx.get(source_name)
            if source_idx is None:
                return None
            out[:, target_idx] = x[:, source_idx].astype(np.float32)

        self._warn_once(
            key="cache.adapt.20to19",
            message=(
                "[WARN] 检测到 20 通道缓存，已按 UAV_TCN_Fine-turn 的 19 通道布局自动适配；"
                "其中 vel_x/vel_y/vel_z 由位置通道差分近似。"
            ),
        )
        return out

    def _load_metadata_from_csv(self, csv_path: Path, expected_length: int) -> Tuple[np.ndarray, int]:
        """
        从 CSV 读取元信息最小子集（fault_state 与 Timestamp）。
        维度约束：
        - 输出 `fault_state`: `[T]`，长度对齐到 `expected_length`；
        - 输出 `sample_rate_hz`: 标量 `int`。
        """

        default_hz = int(getattr(CFG, "windows_sample_rate_hz", 120))

        try:
            header = pd.read_csv(csv_path, nrows=0)
            columns = set(header.columns)
        except Exception:
            return np.zeros((expected_length,), dtype=np.int64), default_hz

        usecols: List[str] = []
        has_fault = "UAVState_data_fault_state" in columns
        has_timestamp = "Timestamp" in columns
        if has_fault:
            usecols.append("UAVState_data_fault_state")
        if has_timestamp:
            usecols.append("Timestamp")

        if len(usecols) == 0:
            return np.zeros((expected_length,), dtype=np.int64), default_hz

        try:
            meta_df = pd.read_csv(csv_path, usecols=usecols)
        except Exception:
            return np.zeros((expected_length,), dtype=np.int64), default_hz

        if has_fault:
            fault_values = pd.to_numeric(meta_df["UAVState_data_fault_state"], errors="coerce").fillna(0.0)
            fault_state = fault_values.to_numpy(dtype=np.int64)
        else:
            fault_state = np.zeros((expected_length,), dtype=np.int64)

        fault_state = self._align_vector_length(
            values=fault_state,
            expected_length=expected_length,
            fill_value=0,
        )

        if has_timestamp:
            ts = pd.to_numeric(meta_df["Timestamp"], errors="coerce").to_numpy(dtype=np.float64)
            sample_rate_hz = self._estimate_sample_rate_from_timestamp(ts=ts, default_hz=default_hz)
        else:
            sample_rate_hz = default_hz

        return fault_state, sample_rate_hz

    def _extract_channels(self, df: pd.DataFrame) -> np.ndarray:
        """
        从 DataFrame 提取输入通道。
        维度变换：
        - 输入 DataFrame: `[T, many_columns]`
        - 输出 `data`: `[T, C]`
        """

        data_list: List[np.ndarray] = []
        for channel in CFG.channels:
            col_data = None
            for candidate in channel.candidates:
                if candidate in df.columns:
                    col_data = df[candidate].to_numpy(dtype=np.float32)
                    break
            if col_data is None:
                print(f"[WARN] 找不到通道 `{channel.name}`，使用 0 填充。")
                col_data = np.zeros((len(df),), dtype=np.float32)
            data_list.append(col_data)
        return np.stack(data_list, axis=1).astype(np.float32)

    @staticmethod
    def _extract_fault_state(df: pd.DataFrame) -> np.ndarray:
        """
        读取故障状态序列。
        输出维度：
        - `fault_state`: `[T]`
        """

        if "UAVState_data_fault_state" not in df.columns:
            return np.zeros((len(df),), dtype=np.int64)

        values = pd.to_numeric(df["UAVState_data_fault_state"], errors="coerce").fillna(0.0)
        return values.to_numpy(dtype=np.int64)

    @staticmethod
    def _compute_fault_onset_idx(fault_state: np.ndarray) -> int:
        """
        计算故障注入起点：`fault_state != 0` 的首个位置。
        若不存在故障，则返回 -1。
        """

        non_zero = np.where(np.asarray(fault_state).reshape(-1) != 0)[0]
        if len(non_zero) == 0:
            return -1
        return int(non_zero[0])

    @staticmethod
    def _estimate_sample_rate_from_timestamp(ts: np.ndarray, default_hz: int) -> int:
        """
        由 Timestamp 序列估计采样率。
        支持三种常见单位：
        - 秒（s）
        - 毫秒（ms）
        - 微秒（us）
        """

        if len(ts) < 2:
            return int(default_hz)

        diffs = np.diff(ts)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs) == 0:
            return int(default_hz)

        dt = float(np.median(diffs))
        if dt >= 1000.0:
            hz = int(round(1_000_000.0 / dt))  # 微秒间隔
        elif dt >= 1.0:
            hz = int(round(1_000.0 / dt))  # 毫秒间隔
        else:
            hz = int(round(1.0 / dt))  # 秒间隔
        if hz <= 0:
            return int(default_hz)
        return hz

    @classmethod
    def _estimate_sample_rate_hz(cls, df: pd.DataFrame) -> int:
        """
        从 DataFrame 估计采样率（兼容旧调用路径）。
        """

        default_hz = int(getattr(CFG, "windows_sample_rate_hz", 120))
        if "Timestamp" not in df.columns:
            return default_hz
        ts = pd.to_numeric(df["Timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        return cls._estimate_sample_rate_from_timestamp(ts=ts, default_hz=default_hz)

    @staticmethod
    def _align_vector_length(values: np.ndarray, expected_length: int, fill_value: int = 0) -> np.ndarray:
        """
        将一维向量长度对齐到 `expected_length`。
        用途：
        - 防止缓存数据长度与 CSV 元信息长度略有差异时触发下游索引错误。
        """

        arr = np.asarray(values).reshape(-1)
        target = int(max(expected_length, 0))
        if len(arr) == target:
            return arr.astype(np.int64, copy=False)
        if len(arr) > target:
            return arr[:target].astype(np.int64, copy=False)
        out = np.full((target,), int(fill_value), dtype=np.int64)
        if len(arr) > 0:
            out[: len(arr)] = arr.astype(np.int64, copy=False)
        return out

    def _warn_once(self, key: str, message: str) -> None:
        """
        同类警告仅打印一次，避免大数据集训练时日志被刷屏。
        """

        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        print(message)


def compute_zscore_stats(
    records: List[Dict],
    loader: MissionLoader,
    mode: str = "sampled",
    max_missions: int = 50,
    window_lengths: Tuple[int, ...] = (60, 120, 240),
    windows_per_mission_per_scale: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Z-score 标准化统计量。
    返回：
    - `mean`: `[C]`
    - `std`: `[C]`
    """

    rng = np.random.default_rng(seed)

    if mode == "full":
        all_data = [loader.load(record["file_path"]) for record in records]
        concatenated = np.concatenate(all_data, axis=0)
        mean = np.mean(concatenated, axis=0).astype(np.float32)
        std = np.std(concatenated, axis=0).astype(np.float32)
        std = np.maximum(std, 1e-8)
        return mean, std

    sampled_records = records
    if len(records) > max_missions:
        indices = rng.choice(len(records), size=max_missions, replace=False)
        sampled_records = [records[i] for i in indices]

    all_windows: List[np.ndarray] = []
    for record in sampled_records:
        data = loader.load(record["file_path"])
        t_len = len(data)
        for wlen in window_lengths:
            if t_len < wlen:
                continue
            for _ in range(windows_per_mission_per_scale):
                start = int(rng.integers(0, t_len - wlen + 1))
                all_windows.append(data[start : start + wlen])

    if len(all_windows) == 0:
        return compute_zscore_stats(
            records=records,
            loader=loader,
            mode="full",
            max_missions=max_missions,
            window_lengths=window_lengths,
            windows_per_mission_per_scale=windows_per_mission_per_scale,
            seed=seed,
        )

    concatenated = np.concatenate(all_windows, axis=0)
    mean = np.mean(concatenated, axis=0).astype(np.float32)
    std = np.std(concatenated, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-8)
    return mean, std
