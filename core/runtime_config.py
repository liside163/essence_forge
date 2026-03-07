"""
config.py

集中加载并校验 `config.yaml`（当前文件使用 JSON 语法）。
所有可调参数统一通过全局 `CFG` 暴露。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from essence_forge.core.channel_layout import (
    channel_names_from_specs,
    validate_legacy_core_prefix,
    validate_unique_channel_names,
)


CORE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CORE_DIR.parent
DEFAULT_CONFIG_FILENAME = "simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json"
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "configs" / DEFAULT_CONFIG_FILENAME
CONFIG_ENV_KEYS = ("ESSENCE_FORGE_CONFIG_PATH", "UAV_TCN_CONFIG_PATH")


@dataclass(frozen=True)
class ChannelSpec:
    """
    输入通道定义。
    - `name`: 规范通道名
    - `candidates`: CSV 候选列名列表
    """

    name: str
    candidates: Tuple[str, ...]


@dataclass(frozen=True)
class Config:
    """
    配置容器。
    采用 dataclass + 字典委托，兼容 `CFG.xxx` 属性访问。
    """

    values: Dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        raise AttributeError(f"Config has no attribute {name!r}")

    def as_dict(self) -> Dict[str, Any]:
        """返回浅拷贝字典，便于日志与快照导出。"""

        return dict(self.values)


def _config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """
    获取配置路径。
    优先读取环境变量 `UAV_TCN_CONFIG_PATH`，未设置时使用当前目录 `config.yaml`。
    """

    if path is not None:
        return Path(path).expanduser().resolve()

    for env_key in CONFIG_ENV_KEYS:
        env_path = os.environ.get(env_key, "").strip()
        if env_path:
            return Path(env_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH.resolve()


def _read_yaml_json(path: Path) -> Dict[str, Any]:
    """
    读取配置文件（JSON 语法）。
    """

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置解析失败（需 JSON 语法）: {path}\n{exc}") from exc


def _require(d: Dict[str, Any], key: str) -> Any:
    """
    读取必填键，缺失时抛出明确异常。
    """

    if key not in d:
        raise KeyError(f"config.yaml 缺少必填字段: {key}")
    return d[key]


def _resolve_path(base: Path, value: str) -> Path:
    """
    将路径规范化为绝对路径。
    - 非 Windows 环境下，将 `D:/...` 映射到 `/mnt/d/...`
    - 相对路径按 `base` 解析
    """

    value_str = str(value)
    if len(value_str) >= 3 and value_str[1] == ":" and value_str[2] in ("/", "\\"):
        if os.name != "nt":
            drive = value_str[0].lower()
            tail = value_str[3:].replace("\\", "/")
            return Path(f"/mnt/{drive}/{tail}").resolve()
        return Path(value_str).resolve()

    p = Path(value_str)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _parse_channels(channels_raw: List[Dict[str, Any]]) -> Tuple[ChannelSpec, ...]:
    """
    解析通道配置列表。
    每个通道必须包含 `name` 与非空 `candidates`。
    """

    channels: List[ChannelSpec] = []
    for item in channels_raw:
        name = str(_require(item, "name"))
        candidates_raw = _require(item, "candidates")
        if not isinstance(candidates_raw, list) or len(candidates_raw) == 0:
            raise ValueError(f"通道 `{name}` 的 candidates 必须是非空列表")
        channels.append(ChannelSpec(name=name, candidates=tuple(str(x) for x in candidates_raw)))
    return tuple(channels)


def _parse_int_key_dict(raw: Dict[str, Any]) -> Dict[int, Any]:
    """将字符串键字典转为整数键字典。"""

    return {int(k): v for k, v in raw.items()}


def _as_tuple_int(values: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    """将可迭代数据解析为 `Tuple[int, ...]`。"""

    if values is None:
        return default
    return tuple(int(v) for v in values)


def _as_tuple_float(values: Any, default: Tuple[float, ...]) -> Tuple[float, ...]:
    """将可迭代数据解析为 `Tuple[float, ...]`。"""

    if values is None:
        return default
    return tuple(float(v) for v in values)


def _validate_prob(name: str, value: float) -> None:
    """校验概率在 [0, 1]。"""

    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} 必须在 [0,1] 范围，当前={value}")


def _validate_pos(name: str, value: float, allow_zero: bool = False) -> None:
    """校验正数/非负数。"""

    v = float(value)
    if allow_zero:
        if v < 0.0:
            raise ValueError(f"{name} 必须 >= 0，当前={value}")
    else:
        if v <= 0.0:
            raise ValueError(f"{name} 必须 > 0，当前={value}")


def _normalize_threshold_search_objective(value: Any) -> str:
    """
    规范化阈值搜索目标，避免配置中别名导致行为不一致。
    """
    raw = str(value or "accuracy_first_lexicographic").strip().lower()
    if raw in {
        "accuracy_first_lexicographic",
        "accuracy_first",
        "accuracy_first_lexicographic(acc,macro_f1)",
    }:
        return "accuracy_first_lexicographic"
    if raw in {
        "macro_f1_with_accuracy_floor",
        "macro_f1_first",
        "macro_f1_priority",
    }:
        return "macro_f1_with_accuracy_floor"
    raise ValueError(
        "fine_tune.threshold_search_objective 必须是 "
        "accuracy_first_lexicographic/macro_f1_with_accuracy_floor"
    )


def _validate_class_ids(name: str, values: Tuple[int, ...], valid_set: set[int], allow_empty: bool = False) -> None:
    """校验类 ID 列表合法性。"""

    if not allow_empty and len(values) == 0:
        raise ValueError(f"{name} 不能为空")
    for cid in values:
        if int(cid) not in valid_set:
            raise ValueError(f"{name} 含非法类 ID={cid}，合法范围={sorted(valid_set)}")


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """
    加载并校验配置，返回只读配置对象。
    """

    cfg_path = _config_path(path)
    raw = _read_yaml_json(cfg_path)
    cfg_dir = cfg_path.parent

    paths = _require(raw, "paths")
    cache = raw.get("cache", {})
    source = _require(raw, "source_domain")
    target = _require(raw, "target_domain")
    dataset = _require(raw, "dataset")
    windows = _require(raw, "windows")
    train = _require(raw, "train")
    fine_tune = _require(raw, "fine_tune")
    model = _require(raw, "model")
    augment = _require(raw, "augment")
    targeted_gan = augment.get("targeted_gan", {})
    imbalance = _require(raw, "imbalance")
    stats = _require(raw, "stats")
    logging_cfg = _require(raw, "logging")
    labels = _require(raw, "labels")
    cross_sensor_residuals = raw.get("cross_sensor_residuals", {})
    if not isinstance(cross_sensor_residuals, dict):
        raise ValueError("cross_sensor_residuals 必须是对象")
    sampler_class_boost_raw = train.get("sampler_class_boost_map", {})
    if not isinstance(sampler_class_boost_raw, dict):
        raise ValueError("train.sampler_class_boost_map 必须是对象映射")

    project_dir = _resolve_path(cfg_dir, str(_require(paths, "project_dir")))
    data_dir = _resolve_path(project_dir, str(_require(paths, "data_dir")))
    outputs_dir = _resolve_path(project_dir, str(_require(paths, "outputs_dir")))

    channels_payload = raw.get("channels", raw.get("channels_19"))
    if channels_payload is None:
        raise KeyError("config.yaml 缺少必填字段: channels (或兼容字段 channels_19)")
    channels = _parse_channels(channels_payload)
    channel_names = channel_names_from_specs(channels)
    validate_unique_channel_names(channel_names)
    input_dim = len(channels)
    residual_channels_raw = cross_sensor_residuals.get("channels", [])
    residual_channel_specs: List[Dict[str, Any]] = (
        [x for x in residual_channels_raw if isinstance(x, dict)]
        if isinstance(residual_channels_raw, list)
        else []
    )
    residual_channel_names = tuple(
        str(item.get("name", f"res_{idx}"))
        for idx, item in enumerate(residual_channel_specs)
    )
    cross_sensor_residual_channels = (
        len(residual_channel_specs)
        if len(residual_channel_specs) > 0
        else int(cross_sensor_residuals.get("num_channels", 9))
    )

    fault_to_class = {
        int(k): int(v) for k, v in _parse_int_key_dict(_require(labels, "fault_to_class")).items()
    }
    class_id_to_name = {
        int(k): str(v) for k, v in _parse_int_key_dict(_require(labels, "class_id_to_name")).items()
    }

    targeted_gan_reference_eval_json_raw = str(targeted_gan.get("reference_eval_json", "")).strip()
    targeted_gan_reference_eval_json = (
        str(_resolve_path(project_dir, targeted_gan_reference_eval_json_raw))
        if targeted_gan_reference_eval_json_raw
        else ""
    )
    targeted_gan_cache_dir_raw = str(targeted_gan.get("cache_dir", "")).strip()
    targeted_gan_cache_dir = (
        str(_resolve_path(project_dir, targeted_gan_cache_dir_raw))
        if targeted_gan_cache_dir_raw
        else ""
    )

    values: Dict[str, Any] = {
        "project_dir": project_dir,
        "data_dir": data_dir,
        "outputs_dir": outputs_dir,
        "cache_enabled": bool(cache.get("enabled", False)),
        "cache_dir": _resolve_path(project_dir, str(cache.get("cache_dir", "./cache_npy"))),
        "cache_format": str(cache.get("format", "npy")).lower().strip(),
        "cache_mmap": bool(cache.get("mmap", False)),
        "source_A": _as_tuple_int(_require(source, "A"), ()),
        "source_B": _as_tuple_int(_require(source, "B"), ()),
        "source_faults": _as_tuple_int(_require(source, "use_faults"), ()),
        "target_A": _as_tuple_int(_require(target, "A"), ()),
        "target_B": _as_tuple_int(_require(target, "B"), ()),
        "target_faults": _as_tuple_int(_require(target, "use_faults"), ()),
        "split_seed": int(_require(dataset, "split_seed")),
        "run_seed": int(raw.get("seed", _require(dataset, "split_seed"))),
        "train_ratio": float(_require(dataset, "train_ratio")),
        "val_ratio": float(_require(dataset, "val_ratio")),
        "test_ratio": float(_require(dataset, "test_ratio")),
        "stratify_by_fault": bool(_require(dataset, "stratify_by_fault")),
        "windows_enable_stage_windowing": bool(windows.get("enable_stage_windowing", False)),
        "windows_sample_rate_hz": int(windows.get("sample_rate_hz", 120)),
        "windows_stage_window_seconds": float(windows.get("stage_window_seconds", 3.0)),
        "windows_stage_stride_seconds": float(windows.get("stage_stride_seconds", 1.0)),
        "windows_normal_context_seconds": float(windows.get("normal_context_seconds", 3.0)),
        "windows_drop_prefault_normal_windows_for_fault_missions": bool(
            windows.get("drop_prefault_normal_windows_for_fault_missions", False)
        ),
        "windows_stage_label_policy": str(windows.get("stage_label_policy", "pre_normal_to_class10_post_to_fault")),
        "window_lengths": _as_tuple_int(_require(windows, "window_lengths"), ()),
        "train_windows_per_mission_per_scale": int(_require(windows, "train_windows_per_mission_per_scale")),
        "eval_windows_per_scale": int(_require(windows, "eval_windows_per_scale")),
        "channels": channels,
        "channel_names": channel_names,
        "input_dim": input_dim,
        "use_cross_sensor_residuals": bool(cross_sensor_residuals.get("enable", False)),
        "cross_sensor_residual_channels": int(cross_sensor_residual_channels),
        "cross_sensor_residual_channel_names": residual_channel_names,
        "cross_sensor_residual_window_norm": bool(cross_sensor_residuals.get("window_norm", True)),
        "cross_sensor_residual_norm_eps": float(cross_sensor_residuals.get("norm_eps", 1e-6)),
        "cross_sensor_residual_mask_mode": str(
            cross_sensor_residuals.get("mask_mode", "base_only")
        ).strip().lower(),
        "device": str(_require(train, "device")),
        "num_workers": int(_require(train, "num_workers")),
        "batch_size": int(_require(train, "batch_size")),
        "sampler_mode": str(train.get("sampler_mode", "none")),
        "sampler_class_boost_map": {
            int(k): float(v)
            for k, v in _parse_int_key_dict(sampler_class_boost_raw).items()
        },
        "normal_class_downsample_ratio": float(train.get("normal_class_downsample_ratio", 1.0)),
        "pin_memory": bool(_require(train, "pin_memory")),
        "persistent_workers": bool(_require(train, "persistent_workers")),
        "prefetch_factor": int(_require(train, "prefetch_factor")),
        "use_amp": bool(_require(train, "use_amp")),
        "learning_rate": float(_require(train, "learning_rate")),
        "weight_decay": float(train.get("weight_decay", 0.0)),
        "max_epochs": int(_require(train, "max_epochs")),
        "early_stopping_patience": int(_require(train, "early_stopping_patience")),
        "scheduler_type": str(train.get("scheduler", {}).get("type", "none")),
        "scheduler_T_0": int(train.get("scheduler", {}).get("T_0", 5)),
        "scheduler_T_mult": int(train.get("scheduler", {}).get("T_mult", 2)),
        "scheduler_eta_min": float(train.get("scheduler", {}).get("eta_min", 1e-6)),
        "val_interval_epochs": int(train.get("val_interval_epochs", 1)),
        "fail_on_nan": bool(train.get("fail_on_nan", True)),
        "ft_freeze_layers": int(_require(fine_tune, "freeze_layers")),
        "ft_learning_rate": float(_require(fine_tune, "learning_rate")),
        "ft_lr_backbone": float(fine_tune.get("lr_backbone", _require(fine_tune, "learning_rate"))),
        "ft_lr_head": float(fine_tune.get("lr_head", _require(fine_tune, "learning_rate"))),
        "ft_weight_decay": float(fine_tune.get("weight_decay", 0.0)),
        "ft_max_epochs": int(_require(fine_tune, "max_epochs")),
        "ft_freeze_epochs": int(fine_tune.get("freeze_epochs", 0)),
        "ft_early_stopping_patience": int(_require(fine_tune, "early_stopping_patience")),
        "ft_early_stopping_warmup_epochs": int(
            fine_tune.get("ft_early_stopping_warmup_epochs", 8)
        ),
        "ft_early_stopping_metric": str(
            fine_tune.get("ft_early_stopping_metric", "macro_f1")
        ).strip().lower(),
        "checkpoint_top_k": int(fine_tune.get("checkpoint_top_k", 3)),
        "last_checkpoint_averaging": bool(
            fine_tune.get("last_checkpoint_averaging", False)
        ),
        "ft_scheduler_type": str(fine_tune.get("scheduler", {}).get("type", "none")),
        "ft_scheduler_T_0": int(fine_tune.get("scheduler", {}).get("T_0", 3)),
        "ft_scheduler_T_mult": int(fine_tune.get("scheduler", {}).get("T_mult", 2)),
        "ft_scheduler_eta_min": float(fine_tune.get("scheduler", {}).get("eta_min", 1e-6)),
        "ft_val_interval_epochs": int(fine_tune.get("val_interval_epochs", 1)),
        "ft_fail_on_nan": bool(fine_tune.get("fail_on_nan", True)),
        "ft_augment_mode": str(fine_tune.get("augment_mode", "full")).strip().lower(),
        "threshold_search": bool(_require(fine_tune, "threshold_search")),
        "threshold_search_mode": str(
            fine_tune.get("threshold_search_mode", "auto")
        ).strip().lower(),
        "threshold_search_max_accuracy_drop": float(
            fine_tune.get("threshold_search_max_accuracy_drop", 0.0)
        ),
        "threshold_search_class10_constraint": str(
            fine_tune.get("threshold_search_class10_constraint", "soft")
        ).strip().lower(),
        "threshold_search_require_non_decreasing_macro_f1": bool(
            fine_tune.get("threshold_search_require_non_decreasing_macro_f1", True)
        ),
        "threshold_search_class10_overpredict_limit": float(
            fine_tune.get("threshold_search_class10_overpredict_limit", 0.0)
        ),
        "threshold_search_min_macro_recall_delta": float(
            fine_tune.get("threshold_search_min_macro_recall_delta", -0.01)
        ),
        "threshold_search_min_gmean_delta": float(
            fine_tune.get("threshold_search_min_gmean_delta", -0.02)
        ),
        "threshold_search_accuracy_tie_eps": float(
            fine_tune.get("threshold_search_accuracy_tie_eps", 0.0015)
        ),
        "threshold_search_bootstrap_rounds": int(
            fine_tune.get("threshold_search_bootstrap_rounds", 200)
        ),
        "threshold_search_objective": _normalize_threshold_search_objective(
            fine_tune.get("threshold_search_objective", "accuracy_first_lexicographic")
        ),
        "threshold_temperature": float(fine_tune.get("threshold_temperature", 1.0)),
        "ft_adabn_enable": bool(fine_tune.get("adabn_enable", False)),
        "ft_adabn_split": str(fine_tune.get("adabn_split", "val")).strip().lower(),
        "ft_adabn_max_batches": int(fine_tune.get("adabn_max_batches", 0)),
        "num_classes": int(_require(model, "num_classes")),
        "tcn_num_levels": int(_require(model, "tcn_num_levels")),
        "tcn_channels": int(_require(model, "tcn_channels")),
        "tcn_kernel_size": int(_require(model, "tcn_kernel_size")),
        "tcn_dropout": float(_require(model, "tcn_dropout")),
        "use_lwpt_frontend": bool(model.get("use_lwpt_frontend", False)),
        "lwpt_num_bands": int(model.get("lwpt_num_bands", 4)),
        "lwpt_kernel_sizes": _as_tuple_int(model.get("lwpt_kernel_sizes", [3, 5, 7, 9]), (3, 5, 7, 9)),
        "lwpt_dropout": float(model.get("lwpt_dropout", 0.1)),
        "use_channel_se": bool(model.get("use_channel_se", True)),
        "channel_se_reduction": int(model.get("channel_se_reduction", 8)),
        "use_mixed_pooling": bool(model.get("use_mixed_pooling", True)),
        "freq_branch_channels": int(model.get("freq_branch_channels", model.get("tcn_channels", 64))),
        "freq_branch_kernel_size": int(model.get("freq_branch_kernel_size", 3)),
        "freq_branch_dropout": float(model.get("freq_branch_dropout", model.get("tcn_dropout", 0.1))),
        "use_freq_branch_se": bool(model.get("use_freq_branch_se", model.get("use_channel_se", True))),
        "freq_branch_se_reduction": int(model.get("freq_branch_se_reduction", model.get("channel_se_reduction", 8))),
        "concat_health_mask_channels": bool(model.get("concat_health_mask_channels", True)),
        "use_sensor_group_attention": bool(model.get("use_sensor_group_attention", False)),
        "use_fused_amtc_tcn": bool(model.get("use_fused_amtc_tcn", False)),
        "fused_tcn_kernel_sizes": _as_tuple_int(model.get("fused_tcn_kernel_sizes", [3, 5, 7]), (3, 5, 7)),
        "fused_tcn_use_pool_branch": bool(model.get("fused_tcn_use_pool_branch", True)),
        "fused_tcn_fft_temperature": float(model.get("fused_tcn_fft_temperature", 2.0)),
        "fused_tcn_use_fft_adaptive": bool(model.get("fused_tcn_use_fft_adaptive", False)),
        "use_deformable_tcn": bool(model.get("use_deformable_tcn", False)),
        "deformable_conv_mode": str(model.get("deformable_conv_mode", "conv1_only")).strip().lower(),
        "deformable_causal_mode": str(model.get("deformable_causal_mode", "strict")).strip().lower(),
        "deform_offset_kernel_size": int(model.get("deform_offset_kernel_size", 3)),
        "deform_offset_l1_weight": float(model.get("deform_offset_l1_weight", 1e-4)),
        "use_hybrid_deform_multiscale_tcn": bool(model.get("use_hybrid_deform_multiscale_tcn", False)),
        "hybrid_tcn_kernel_sizes": _as_tuple_int(model.get("hybrid_tcn_kernel_sizes", [3, 5, 7]), (3, 5, 7)),
        "use_levelwise_tcn_aggregation": bool(model.get("use_levelwise_tcn_aggregation", False)),
        "use_uncertainty_temporal_pooling": bool(model.get("use_uncertainty_temporal_pooling", False)),
        "use_tf2d_branch": bool(_require(model, "use_tf2d_branch")),
        "tf2d_input_from_raw": bool(_require(model, "tf2d_input_from_raw")),
        "tf2d_n_fft": _as_tuple_int(_require(model, "tf2d_n_fft"), ()),
        "tf2d_hop_lengths": _as_tuple_int(_require(model, "tf2d_hop_lengths"), ()),
        "tf2d_win_lengths": _as_tuple_int(_require(model, "tf2d_win_lengths"), ()),
        "tf2d_log_scale": bool(_require(model, "tf2d_log_scale")),
        "tf2d_branch_channels": int(_require(model, "tf2d_branch_channels")),
        "tf2d_pool_freq_bins": int(_require(model, "tf2d_pool_freq_bins")),
        "tf2d_pool_time_bins": int(_require(model, "tf2d_pool_time_bins")),
        "use_layerwise_tf_fusion": bool(_require(model, "use_layerwise_tf_fusion")),
        "tf_fusion_alpha_init": float(_require(model, "tf_fusion_alpha_init")),
        "tf_fusion_dropout": float(_require(model, "tf_fusion_dropout")),
        "use_global_branch_weighting": bool(_require(model, "use_global_branch_weighting")),
        "fft_mag_log1p": bool(model.get("fft_mag_log1p", True)),
        "fft_mag_norm": str(model.get("fft_mag_norm", "per_sample_channel")).strip().lower(),
        "enable_augment": bool(_require(augment, "enable_augment")),
        "augment_strategy": str(augment.get("strategy", "legacy")),
        "augment_disable_targeted_gan": bool(augment.get("disable_targeted_gan", False)),
        "augment_gaussian_sigma": float(_require(augment, "gaussian_sigma")),
        "enable_drift_noise": bool(_require(augment, "enable_drift_noise")),
        "augment_drift_sigma": float(_require(augment, "drift_sigma")),
        "use_mixup": bool(_require(augment, "use_mixup")),
        "mixup_alpha": float(_require(augment, "mixup_alpha")),
        "augment_correlated_jitter_enable": bool(augment.get("correlated_jitter", {}).get("enable", True)),
        "augment_correlated_jitter_prob": float(augment.get("correlated_jitter", {}).get("prob", 0.7)),
        "augment_correlated_jitter_shared_sigma": float(augment.get("correlated_jitter", {}).get("shared_sigma", 0.015)),
        "augment_correlated_jitter_ind_sigma": float(augment.get("correlated_jitter", {}).get("ind_sigma", 0.03)),
        "augment_bias_injection_enable": bool(augment.get("bias_injection", {}).get("enable", True)),
        "augment_bias_injection_prob": float(augment.get("bias_injection", {}).get("prob", 0.4)),
        "augment_bias_injection_imu_ratio": float(augment.get("bias_injection", {}).get("imu_ratio", 0.05)),
        "augment_bias_injection_posvel_ratio": float(augment.get("bias_injection", {}).get("posvel_ratio", 0.02)),
        "augment_linear_drift_enable": bool(augment.get("linear_drift", {}).get("enable", True)),
        "augment_linear_drift_prob": float(augment.get("linear_drift", {}).get("prob", 0.35)),
        "augment_linear_drift_range": _as_tuple_float(augment.get("linear_drift", {}).get("range", [0.001, 0.01]), (0.001, 0.01)),
        "augment_gain_scaling_enable": bool(augment.get("gain_scaling", {}).get("enable", True)),
        "augment_gain_scaling_prob": float(augment.get("gain_scaling", {}).get("prob", 0.5)),
        "augment_gain_scaling_imu_range": _as_tuple_float(augment.get("gain_scaling", {}).get("imu_range", [0.97, 1.03]), (0.97, 1.03)),
        "augment_gain_scaling_posvel_range": _as_tuple_float(augment.get("gain_scaling", {}).get("posvel_range", [0.98, 1.02]), (0.98, 1.02)),
        "augment_freq_domain_enable": bool(augment.get("freq_domain", {}).get("enable", False)),
        "augment_freq_time_mask_enable": bool(augment.get("freq_domain", {}).get("time_mask", {}).get("enable", True)),
        "augment_freq_time_mask_prob": float(augment.get("freq_domain", {}).get("time_mask", {}).get("prob", 0.35)),
        "augment_freq_time_mask_width_range": _as_tuple_int(augment.get("freq_domain", {}).get("time_mask", {}).get("width_range", [2, 6]), (2, 6)),
        "augment_freq_freq_mask_enable": bool(augment.get("freq_domain", {}).get("freq_mask", {}).get("enable", True)),
        "augment_freq_freq_mask_prob": float(augment.get("freq_domain", {}).get("freq_mask", {}).get("prob", 0.35)),
        "augment_freq_freq_mask_width_range": _as_tuple_int(augment.get("freq_domain", {}).get("freq_mask", {}).get("width_range", [1, 2]), (1, 2)),
        "augment_freq_spectral_tilt_enable": bool(augment.get("freq_domain", {}).get("spectral_tilt", {}).get("enable", True)),
        "augment_freq_spectral_tilt_prob": float(augment.get("freq_domain", {}).get("spectral_tilt", {}).get("prob", 0.2)),
        "augment_freq_spectral_tilt_db_range": float(augment.get("freq_domain", {}).get("spectral_tilt", {}).get("db_range", 1.5)),
        "augment_hard_class_targets": _as_tuple_int(augment.get("hard_class", {}).get("target_classes", [5, 6, 7]), (5, 6, 7)),
        "augment_hard_class_min_prob": float(augment.get("hard_class", {}).get("min_prob", 0.8)),
        "augment_sensor_fault_bias_mult": float(augment.get("hard_class", {}).get("sensor_fault", {}).get("bias_mult", 8.0)),
        "augment_sensor_fault_stuck_prob": float(augment.get("hard_class", {}).get("sensor_fault", {}).get("stuck_prob", 0.3)),
        "augment_sensor_fault_stuck_range": _as_tuple_int(augment.get("hard_class", {}).get("sensor_fault", {}).get("stuck_range", [30, 100]), (30, 100)),
        "augment_hard_class_mixup_enable": bool(augment.get("hard_class", {}).get("mixup", {}).get("enable", True)),
        "augment_hard_class_mixup_prob": float(augment.get("hard_class", {}).get("mixup", {}).get("prob", 0.35)),
        "augment_hard_class_mixup_alpha": float(augment.get("hard_class", {}).get("mixup", {}).get("alpha", 0.3)),
        "augment_class_targeted_enable": bool(augment.get("class_targeted", {}).get("enable", True)),
        "augment_class_targeted_focus_classes": _as_tuple_int(augment.get("class_targeted", {}).get("focus_classes", [2, 4, 8, 9]), (2, 4, 8, 9)),
        "augment_class_targeted_min_prob": float(augment.get("class_targeted", {}).get("min_prob", 0.45)),
        "augment_class_targeted_max_prob": float(augment.get("class_targeted", {}).get("max_prob", 0.90)),
        "augment_confusion_pair_enable": bool(augment.get("confusion_pair", {}).get("enable", True)),
        "augment_confusion_pair_pairs": tuple(
            tuple(int(v) for v in pair)
            for pair in augment.get(
                "confusion_pair",
                {},
            ).get(
                "pairs",
                [
                    [0, 10],
                    [10, 0],
                    [4, 10],
                    [10, 4],
                    [8, 10],
                    [10, 8],
                    [9, 10],
                    [10, 9],
                    [8, 9],
                    [9, 8],
                ],
            )
        ),
        "augment_confusion_pair_pair_boost": float(augment.get("confusion_pair", {}).get("pair_boost", 0.30)),
        "augment_boundary_jitter_enable": bool(augment.get("boundary_jitter", {}).get("enable", True)),
        "augment_boundary_jitter_near_onset_seconds": float(augment.get("boundary_jitter", {}).get("near_onset_seconds", 1.5)),
        "augment_boundary_jitter_augment_prob": float(augment.get("boundary_jitter", {}).get("augment_prob", 0.60)),
        "augment_boundary_jitter_noise_sigma": float(augment.get("boundary_jitter", {}).get("noise_sigma", 0.03)),
        "augment_feature_budget_enable": bool(augment.get("feature_budget", {}).get("enable", True)),
        "augment_feature_budget_protected_channels": tuple(
            str(v) for v in augment.get("feature_budget", {}).get("protected_channels", ["pos_x", "pos_y", "mag_x", "mag_y", "mag_z", "gyro_z", "accel_y", "accel_z"])
        ),
        "augment_feature_budget_protected_sigma_scale": float(augment.get("feature_budget", {}).get("protected_sigma_scale", 0.60)),
        "augment_feature_budget_unprotected_sigma_scale": float(augment.get("feature_budget", {}).get("unprotected_sigma_scale", 1.40)),
        "augment_sensor_dropout_enable": bool(augment.get("sensor_dropout", {}).get("enable", False)),
        "augment_sensor_dropout_prob": float(augment.get("sensor_dropout", {}).get("prob", 0.15)),
        "augment_sensor_dropout_group_drop_prob": float(
            augment.get("sensor_dropout", {}).get("group_drop_prob", 0.30)
        ),
        "augment_sensor_dropout_protected_groups": tuple(
            str(v) for v in augment.get("sensor_dropout", {}).get("protected_groups", ["accel"])
        ),
        "augment_class_mixup_enable": bool(augment.get("class_mixup", {}).get("enable", augment.get("hard_class", {}).get("mixup", {}).get("enable", True))),
        "augment_class_mixup_target_classes": _as_tuple_int(augment.get("class_mixup", {}).get("target_classes", [2, 4, 8, 9]), (2, 4, 8, 9)),
        "augment_class_mixup_prob": float(augment.get("class_mixup", {}).get("prob", augment.get("hard_class", {}).get("mixup", {}).get("prob", 0.25))),
        "augment_class_mixup_alpha": float(augment.get("class_mixup", {}).get("alpha", augment.get("hard_class", {}).get("mixup", {}).get("alpha", 0.15))),
        "augment_class_mixup_same_fault_family_only": bool(augment.get("class_mixup", {}).get("same_fault_family_only", True)),
        "augment_class_mixup_max_bank_size_per_class": int(
            augment.get("class_mixup", {}).get("max_bank_size_per_class", 128)
        ),
        "augment_curriculum_warmup": int(augment.get("curriculum", {}).get("warmup_epochs", 8)),
        "augment_curriculum_full_start": int(augment.get("curriculum", {}).get("full_aug_start_epoch", 8)),
        "augment_curriculum_decay_ratio": float(augment.get("curriculum", {}).get("decay_start_ratio", 0.85)),
        "augment_curriculum_decay_factor": float(augment.get("curriculum", {}).get("decay_factor", 0.5)),
        "targeted_gan_enable": bool(targeted_gan.get("enable", False) and not augment.get("disable_targeted_gan", False)),
        "targeted_gan_reference_eval_json": targeted_gan_reference_eval_json,
        "targeted_gan_manual_hard_classes": _as_tuple_int(targeted_gan.get("manual_hard_classes", []), ()),
        "targeted_gan_top_k": int(targeted_gan.get("top_k", 5)),
        "targeted_gan_exclude_classes": _as_tuple_int(targeted_gan.get("exclude_classes", [10]), (10,)),
        "targeted_gan_min_support": int(targeted_gan.get("min_support", 30)),
        "targeted_gan_max_windows_per_class": int(targeted_gan.get("max_windows_per_class", 512)),
        "targeted_gan_apply_prob": float(targeted_gan.get("apply_prob", 0.8)),
        "targeted_gan_blend_ratio_min": float(targeted_gan.get("blend_ratio_min", 0.35)),
        "targeted_gan_blend_ratio_max": float(targeted_gan.get("blend_ratio_max", 0.65)),
        "targeted_gan_latent_dim": int(targeted_gan.get("latent_dim", 96)),
        "targeted_gan_hidden_dim": int(targeted_gan.get("hidden_dim", 256)),
        "targeted_gan_train_steps": int(targeted_gan.get("train_steps", 180)),
        "targeted_gan_critic_steps": int(targeted_gan.get("critic_steps", 3)),
        "targeted_gan_batch_size": int(targeted_gan.get("batch_size", 64)),
        "targeted_gan_lr_generator": float(targeted_gan.get("lr_generator", 2e-4)),
        "targeted_gan_lr_discriminator": float(targeted_gan.get("lr_discriminator", 4e-4)),
        "targeted_gan_gp_lambda": float(targeted_gan.get("gp_lambda", 10.0)),
        "targeted_gan_spectral_lambda": float(targeted_gan.get("spectral_lambda", 1.0)),
        "targeted_gan_smooth_lambda": float(targeted_gan.get("smooth_lambda", 0.02)),
        "targeted_gan_synth_bank_size": int(targeted_gan.get("synth_bank_size", 512)),
        "targeted_gan_min_windows_per_class": int(targeted_gan.get("min_windows_per_class", 24)),
        "targeted_gan_seed": int(targeted_gan.get("seed", 2026)),
        "targeted_gan_device": str(targeted_gan.get("device", "cpu")),
        "targeted_gan_cache_enable": bool(targeted_gan.get("cache_enable", False)),
        "targeted_gan_cache_dir": targeted_gan_cache_dir,
        "targeted_gan_cache_force_retrain": bool(targeted_gan.get("cache_force_retrain", False)),
        "class_weight_mode": str(_require(imbalance, "class_weight_mode")),
        "inverse_log_offset": float(_require(imbalance, "inverse_log_offset")),
        "use_label_smoothing": bool(_require(imbalance, "use_label_smoothing")),
        "label_smoothing": float(_require(imbalance, "label_smoothing")),
        "use_focal_loss": bool(_require(imbalance, "use_focal_loss")),
        "focal_gamma": float(_require(imbalance, "focal_gamma")),
        "use_adaptive_cost": bool(_require(imbalance, "use_adaptive_cost")),
        "use_ldam_loss": bool(imbalance.get("use_ldam_loss", False)),
        "ldam_max_m": float(imbalance.get("ldam_max_m", 0.5)),
        "ldam_scale": float(imbalance.get("ldam_scale", 30.0)),
        "use_confusion_aware_loss": bool(imbalance.get("use_confusion_aware_loss", False)),
        "confusion_pairs": tuple(
            (
                int(pair[0]),
                int(pair[1]),
                float(pair[2]),
            )
            for pair in imbalance.get("confusion_pairs", [])
        ),
        "lambda_confusion": float(imbalance.get("lambda_confusion", 1.0)),
        "use_supcon_loss": bool(imbalance.get("use_supcon_loss", False)),
        "supcon_temperature": float(imbalance.get("supcon_temperature", 0.1)),
        "supcon_lambda": float(imbalance.get("supcon_lambda", 0.3)),
        "use_center_loss": bool(imbalance.get("use_center_loss", False)),
        "center_loss_lambda": float(imbalance.get("center_loss_lambda", 0.1)),
        "center_loss_lr": float(imbalance.get("center_loss_lr", 0.5)),
        "center_loss_active_classes": _as_tuple_int(
            imbalance.get("center_loss_active_classes", []),
            (),
        ),
        "loss_type": str(imbalance.get("loss_type", "auto")).strip().lower(),
        "use_optuna_cost_result": bool(imbalance.get("use_optuna_cost_result", False)),
        "optuna_cost_result_filename": str(imbalance.get("optuna_cost_result_filename", "optuna_cost_result.json")),
        "apply_optuna_best_hparams": bool(imbalance.get("apply_optuna_best_hparams", False)),
        "stats_mode": str(_require(stats, "mode")),
        "stats_max_missions": int(_require(stats, "max_missions")),
        "stats_windows_per_mission_per_scale": int(_require(stats, "windows_per_mission_per_scale")),
        "log_print_interval_batches": int(_require(logging_cfg, "print_interval_batches")),
        "log_write_jsonl": bool(_require(logging_cfg, "write_jsonl")),
        "fault_to_class": fault_to_class,
        "class_id_to_name": class_id_to_name,
    }

    cfg = Config(values=values)
    valid_class_ids = set(range(int(cfg.num_classes)))
    channel_name_set = {ch.name for ch in cfg.channels}

    if cfg.num_classes != len(cfg.class_id_to_name):
        raise ValueError("num_classes 与 class_id_to_name 数量不一致")
    if cfg.input_dim <= 0 or cfg.input_dim != len(cfg.channels):
        raise ValueError("input_dim 与 channels 不一致")
    if bool(
        cfg.use_cross_sensor_residuals
        or cfg.concat_health_mask_channels
        or str(cfg.augment_strategy).strip().lower() == "physics_aware"
    ):
        validate_legacy_core_prefix(cfg.channel_names)
    if cfg.use_cross_sensor_residuals:
        if cfg.input_dim < 19:
            raise ValueError("启用 cross_sensor_residuals 时 input_dim 至少需要 19")
        if cfg.cross_sensor_residual_channels != 9:
            raise ValueError(
                "当前实现仅支持 9 个 cross_sensor_residuals 通道，"
                f"当前={cfg.cross_sensor_residual_channels}"
            )
    if cfg.cross_sensor_residual_norm_eps <= 0.0:
        raise ValueError("cross_sensor_residuals.norm_eps 必须 > 0")
    if cfg.cross_sensor_residual_mask_mode != "base_only":
        raise ValueError("当前仅支持 cross_sensor_residuals.mask_mode=base_only")
    if abs((cfg.train_ratio + cfg.val_ratio + cfg.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")
    if cfg.run_seed < 0:
        raise ValueError("seed 必须 >= 0")
    if cfg.cache_format not in {"npy", "npz"}:
        raise ValueError("cache.format 必须是 npy 或 npz")
    if cfg.cache_mmap and cfg.cache_format != "npy":
        raise ValueError("cache.mmap=true 时 cache.format 必须是 npy")
    if cfg.prefetch_factor < 1:
        raise ValueError("prefetch_factor 必须 >= 1")
    if cfg.persistent_workers and cfg.num_workers == 0:
        raise ValueError("persistent_workers=true 时 num_workers 必须 > 0")
    if cfg.sampler_mode not in {"none", "weighted"}:
        raise ValueError("sampler_mode 必须是 none 或 weighted")
    for class_id, boost_factor in cfg.sampler_class_boost_map.items():
        if int(class_id) not in valid_class_ids:
            raise ValueError(f"sampler_class_boost_map 含非法类 ID={class_id}")
        if float(boost_factor) <= 0.0:
            raise ValueError(
                f"sampler_class_boost_map[{class_id}] 必须 > 0，当前={boost_factor}"
            )
    if not (0.0 < cfg.normal_class_downsample_ratio <= 1.0):
        raise ValueError("normal_class_downsample_ratio 必须在 (0,1]")
    if cfg.val_interval_epochs < 1:
        raise ValueError("train.val_interval_epochs 必须 >= 1")
    if cfg.ft_val_interval_epochs < 1:
        raise ValueError("fine_tune.val_interval_epochs 必须 >= 1")
    if cfg.ft_freeze_epochs < 0:
        raise ValueError("fine_tune.freeze_epochs 必须 >= 0")
    if cfg.ft_early_stopping_warmup_epochs < 0:
        raise ValueError("fine_tune.ft_early_stopping_warmup_epochs 必须 >= 0")
    if cfg.ft_early_stopping_metric not in {"macro_f1", "macro_f1_gmean_composite"}:
        raise ValueError(
            "fine_tune.ft_early_stopping_metric 必须是 macro_f1/macro_f1_gmean_composite"
        )
    if cfg.checkpoint_top_k < 1:
        raise ValueError("fine_tune.checkpoint_top_k 必须 >= 1")
    if cfg.ft_lr_backbone <= 0.0 or cfg.ft_lr_head <= 0.0:
        raise ValueError("fine_tune.lr_backbone/lr_head 必须 > 0")
    if cfg.loss_type not in {"auto", "ce", "focal", "ldam"}:
        raise ValueError("imbalance.loss_type 必须是 auto/ce/focal/ldam")
    if cfg.lambda_confusion < 0.0:
        raise ValueError("imbalance.lambda_confusion 必须 >= 0")
    for pair in cfg.confusion_pairs:
        if len(pair) != 3:
            raise ValueError(f"imbalance.confusion_pairs 元素必须是 [src,dst,scale]，当前={pair}")
        src_id, dst_id, pair_scale = int(pair[0]), int(pair[1]), float(pair[2])
        if src_id not in valid_class_ids or dst_id not in valid_class_ids:
            raise ValueError(f"imbalance.confusion_pairs 含非法类 ID，当前={pair}")
        if pair_scale < 0.0:
            raise ValueError(f"imbalance.confusion_pairs 的 scale 必须 >= 0，当前={pair}")
    if cfg.supcon_temperature <= 0.0:
        raise ValueError("imbalance.supcon_temperature 必须 > 0")
    if cfg.supcon_lambda < 0.0:
        raise ValueError("imbalance.supcon_lambda 必须 >= 0")
    if cfg.center_loss_lambda < 0.0:
        raise ValueError("imbalance.center_loss_lambda 必须 >= 0")
    if cfg.center_loss_lr <= 0.0:
        raise ValueError("imbalance.center_loss_lr 必须 > 0")
    _validate_class_ids(
        "imbalance.center_loss_active_classes",
        cfg.center_loss_active_classes,
        valid_class_ids,
        allow_empty=True,
    )
    if cfg.ft_augment_mode not in {"full", "light", "off"}:
        raise ValueError("fine_tune.augment_mode 必须是 full/light/off")
    if cfg.threshold_search_mode not in {"auto", "global", "per_class"}:
        raise ValueError(
            "fine_tune.threshold_search_mode 必须是 auto/global/per_class"
        )
    if cfg.threshold_search_max_accuracy_drop < 0.0:
        raise ValueError(
            "fine_tune.threshold_search_max_accuracy_drop 必须 >= 0"
        )
    if cfg.threshold_search_class10_constraint not in {"soft", "hard", "none"}:
        raise ValueError(
            "fine_tune.threshold_search_class10_constraint 必须是 soft/hard/none"
        )
    if cfg.threshold_search_class10_overpredict_limit < 0.0:
        raise ValueError(
            "fine_tune.threshold_search_class10_overpredict_limit 必须 >= 0"
        )
    if cfg.threshold_search_accuracy_tie_eps < 0.0:
        raise ValueError(
            "fine_tune.threshold_search_accuracy_tie_eps 必须 >= 0"
        )
    if cfg.threshold_search_bootstrap_rounds < 0:
        raise ValueError(
            "fine_tune.threshold_search_bootstrap_rounds 必须 >= 0"
        )
    if cfg.threshold_search_objective not in {
        "accuracy_first_lexicographic",
        "macro_f1_with_accuracy_floor",
    }:
        raise ValueError(
            "fine_tune.threshold_search_objective 必须是 "
            "accuracy_first_lexicographic/macro_f1_with_accuracy_floor"
        )
    if cfg.threshold_temperature <= 0.0:
        raise ValueError("fine_tune.threshold_temperature 必须 > 0")
    if cfg.ft_adabn_split not in {"train", "val", "test"}:
        raise ValueError("fine_tune.adabn_split 必须是 train/val/test")
    if cfg.ft_adabn_max_batches < 0:
        raise ValueError("fine_tune.adabn_max_batches 必须 >= 0")
    if cfg.fft_mag_norm not in {"none", "per_sample_channel", "global"}:
        raise ValueError("model.fft_mag_norm 必须是 none/per_sample_channel/global")
    if cfg.deformable_conv_mode not in {"conv1_only", "both"}:
        raise ValueError("model.deformable_conv_mode 必须是 conv1_only 或 both")
    if cfg.deformable_causal_mode not in {"strict", "relaxed"}:
        raise ValueError("model.deformable_causal_mode 必须是 strict 或 relaxed")
    if cfg.deform_offset_kernel_size <= 1 or cfg.deform_offset_kernel_size % 2 == 0:
        raise ValueError("model.deform_offset_kernel_size 必须是大于1的奇数")
    if cfg.deform_offset_l1_weight < 0.0:
        raise ValueError("model.deform_offset_l1_weight 必须 >= 0")
    if len(cfg.hybrid_tcn_kernel_sizes) == 0:
        raise ValueError("model.hybrid_tcn_kernel_sizes 不能为空")
    for kernel in cfg.hybrid_tcn_kernel_sizes:
        if kernel <= 1 or kernel % 2 == 0:
            raise ValueError(
                f"model.hybrid_tcn_kernel_sizes 必须是大于1的奇数，当前={kernel}"
            )
    if cfg.use_hybrid_deform_multiscale_tcn and not cfg.use_deformable_tcn:
        raise ValueError(
            "model.use_hybrid_deform_multiscale_tcn=true 时 model.use_deformable_tcn 必须为 true"
        )

    if len(cfg.window_lengths) == 0 or any(w <= 0 for w in cfg.window_lengths):
        raise ValueError("window_lengths 必须是非空正整数列表")
    if cfg.windows_sample_rate_hz <= 0:
        raise ValueError("windows.sample_rate_hz 必须 > 0")
    if cfg.windows_stage_window_seconds <= 0.0 or cfg.windows_stage_stride_seconds <= 0.0:
        raise ValueError("windows stage window/stride 必须 > 0")
    if cfg.windows_normal_context_seconds < 0.0:
        raise ValueError("windows.normal_context_seconds 必须 >= 0")
    if cfg.windows_stage_label_policy != "pre_normal_to_class10_post_to_fault":
        raise ValueError("windows.stage_label_policy 当前仅支持 pre_normal_to_class10_post_to_fault")

    if cfg.augment_strategy not in {"legacy", "physics_aware"}:
        raise ValueError("augment.strategy 必须是 legacy 或 physics_aware")

    _validate_class_ids("augment.hard_class.target_classes", cfg.augment_hard_class_targets, valid_class_ids)
    _validate_prob("augment.hard_class.min_prob", cfg.augment_hard_class_min_prob)
    _validate_prob("augment.hard_class.sensor_fault.stuck_prob", cfg.augment_sensor_fault_stuck_prob)
    if len(cfg.augment_sensor_fault_stuck_range) != 2:
        raise ValueError("augment.hard_class.sensor_fault.stuck_range 必须是长度2")
    if cfg.augment_sensor_fault_stuck_range[0] <= 0 or cfg.augment_sensor_fault_stuck_range[1] <= 0:
        raise ValueError("augment.hard_class.sensor_fault.stuck_range 元素必须 > 0")
    if cfg.augment_sensor_fault_stuck_range[0] > cfg.augment_sensor_fault_stuck_range[1]:
        raise ValueError("augment.hard_class.sensor_fault.stuck_range[0] 不能大于 [1]")
    _validate_prob("augment.hard_class.mixup.prob", cfg.augment_hard_class_mixup_prob)
    _validate_pos("augment.hard_class.mixup.alpha", cfg.augment_hard_class_mixup_alpha)

    _validate_prob("augment.class_targeted.min_prob", cfg.augment_class_targeted_min_prob)
    _validate_prob("augment.class_targeted.max_prob", cfg.augment_class_targeted_max_prob)
    if cfg.augment_class_targeted_min_prob > cfg.augment_class_targeted_max_prob:
        raise ValueError("augment.class_targeted.min_prob 不能大于 max_prob")
    if cfg.augment_class_targeted_enable:
        _validate_class_ids("augment.class_targeted.focus_classes", cfg.augment_class_targeted_focus_classes, valid_class_ids)

    _validate_prob("augment.confusion_pair.pair_boost", cfg.augment_confusion_pair_pair_boost)
    if cfg.augment_confusion_pair_enable and len(cfg.augment_confusion_pair_pairs) == 0:
        raise ValueError("augment.confusion_pair.pairs 启用时不能为空")
    for pair in cfg.augment_confusion_pair_pairs:
        if len(pair) != 2:
            raise ValueError(f"augment.confusion_pair.pairs 元素必须长度为2，当前={pair}")
        if int(pair[0]) not in valid_class_ids or int(pair[1]) not in valid_class_ids:
            raise ValueError(f"augment.confusion_pair.pairs 含非法类ID，当前={pair}")

    if cfg.augment_boundary_jitter_enable:
        _validate_pos("augment.boundary_jitter.near_onset_seconds", cfg.augment_boundary_jitter_near_onset_seconds)
    _validate_prob("augment.boundary_jitter.augment_prob", cfg.augment_boundary_jitter_augment_prob)
    _validate_pos("augment.boundary_jitter.noise_sigma", cfg.augment_boundary_jitter_noise_sigma, allow_zero=True)

    if cfg.augment_feature_budget_enable and len(cfg.augment_feature_budget_protected_channels) == 0:
        raise ValueError("augment.feature_budget.protected_channels 启用时不能为空")
    unknown_channels = [name for name in cfg.augment_feature_budget_protected_channels if str(name) not in channel_name_set]
    if len(unknown_channels) > 0:
        raise ValueError(f"augment.feature_budget.protected_channels 含未知通道: {unknown_channels}")
    _validate_pos("augment.feature_budget.protected_sigma_scale", cfg.augment_feature_budget_protected_sigma_scale)
    _validate_pos("augment.feature_budget.unprotected_sigma_scale", cfg.augment_feature_budget_unprotected_sigma_scale)
    _validate_prob("augment.sensor_dropout.prob", cfg.augment_sensor_dropout_prob)
    _validate_prob(
        "augment.sensor_dropout.group_drop_prob",
        cfg.augment_sensor_dropout_group_drop_prob,
    )
    valid_sensor_groups = {"accel", "gyro", "mag", "pos", "vel", "quat"}
    unknown_sensor_groups = [
        str(name)
        for name in cfg.augment_sensor_dropout_protected_groups
        if str(name).strip().lower() not in valid_sensor_groups
    ]
    if len(unknown_sensor_groups) > 0:
        raise ValueError(
            "augment.sensor_dropout.protected_groups 含未知分组: "
            f"{unknown_sensor_groups}"
        )

    _validate_prob("augment.class_mixup.prob", cfg.augment_class_mixup_prob)
    _validate_pos("augment.class_mixup.alpha", cfg.augment_class_mixup_alpha)
    if cfg.augment_class_mixup_max_bank_size_per_class < 1:
        raise ValueError("augment.class_mixup.max_bank_size_per_class 必须 >= 1")
    if cfg.augment_class_mixup_enable:
        _validate_class_ids("augment.class_mixup.target_classes", cfg.augment_class_mixup_target_classes, valid_class_ids)

    if cfg.targeted_gan_top_k < 1:
        raise ValueError("targeted_gan.top_k 必须 >= 1")
    if cfg.targeted_gan_min_support < 0:
        raise ValueError("targeted_gan.min_support 必须 >= 0")
    if cfg.targeted_gan_max_windows_per_class < 1:
        raise ValueError("targeted_gan.max_windows_per_class 必须 >= 1")
    _validate_prob("targeted_gan.apply_prob", cfg.targeted_gan_apply_prob)
    _validate_prob("targeted_gan.blend_ratio_min", cfg.targeted_gan_blend_ratio_min)
    _validate_prob("targeted_gan.blend_ratio_max", cfg.targeted_gan_blend_ratio_max)
    if cfg.targeted_gan_blend_ratio_min > cfg.targeted_gan_blend_ratio_max:
        raise ValueError("targeted_gan.blend_ratio_min 不能大于 blend_ratio_max")
    if cfg.targeted_gan_enable and len(cfg.targeted_gan_reference_eval_json.strip()) == 0:
        raise ValueError("targeted_gan.enable=true 时 reference_eval_json 不能为空")
    if cfg.targeted_gan_enable and not Path(cfg.targeted_gan_reference_eval_json).exists():
        raise ValueError(f"targeted_gan.reference_eval_json 不存在: {cfg.targeted_gan_reference_eval_json}")

    if cfg.use_lwpt_frontend:
        if cfg.lwpt_num_bands < 1:
            raise ValueError("lwpt_num_bands 必须 >= 1")
        if len(cfg.lwpt_kernel_sizes) == 0:
            raise ValueError("lwpt_kernel_sizes 不能为空")
        for kernel in cfg.lwpt_kernel_sizes:
            if kernel <= 1 or kernel % 2 == 0:
                raise ValueError(f"lwpt_kernel_sizes 必须是大于1的奇数，当前={kernel}")
        _validate_prob("lwpt_dropout", cfg.lwpt_dropout)
    if cfg.use_channel_se and cfg.channel_se_reduction <= 0:
        raise ValueError("channel_se_reduction 必须 > 0")
    if cfg.freq_branch_channels <= 0:
        raise ValueError("freq_branch_channels 必须 > 0")
    if cfg.freq_branch_kernel_size <= 1 or cfg.freq_branch_kernel_size % 2 == 0:
        raise ValueError("freq_branch_kernel_size 必须是大于1的奇数")
    if not (0.0 <= cfg.freq_branch_dropout < 1.0):
        raise ValueError("freq_branch_dropout 必须在 [0,1) 范围")
    if cfg.use_freq_branch_se and cfg.freq_branch_se_reduction <= 0:
        raise ValueError("freq_branch_se_reduction 必须 > 0")

    if cfg.use_tf2d_branch:
        if not (len(cfg.tf2d_n_fft) == len(cfg.tf2d_hop_lengths) == len(cfg.tf2d_win_lengths)):
            raise ValueError("tf2d_n_fft/tf2d_hop_lengths/tf2d_win_lengths 长度必须一致")
        if len(cfg.tf2d_n_fft) == 0:
            raise ValueError("tf2d_n_fft 不能为空")
        for n_fft, hop, win in zip(cfg.tf2d_n_fft, cfg.tf2d_hop_lengths, cfg.tf2d_win_lengths):
            if n_fft <= 0 or hop <= 0 or win <= 0:
                raise ValueError(f"TF2D 参数必须 > 0，当前 n_fft={n_fft}, hop={hop}, win={win}")
            if win > n_fft:
                raise ValueError(f"tf2d_win_lengths 必须 <= n_fft，当前 win={win}, n_fft={n_fft}")
        _validate_prob("tf_fusion_dropout", cfg.tf_fusion_dropout)
    else:
        if cfg.use_layerwise_tf_fusion or cfg.use_global_branch_weighting:
            raise ValueError("use_tf2d_branch=false 时不可开启 TF 融合权重")

    return cfg


CFG = load_config()


def reload_config(path: str | os.PathLike[str] | None = None) -> Config:
    loaded = load_config(path)
    CFG.values.clear()
    CFG.values.update(loaded.values)
    return CFG
