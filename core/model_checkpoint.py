"""
model_checkpoint.py

模型 checkpoint 的保存/加载辅助工具。

目标：
1. 训练阶段保存可复现的模型初始化参数（model_init_kwargs + model_class_name）。
2. 加载阶段优先使用 checkpoint 自带参数，避免受当前 config 变更影响。
3. 对历史 checkpoint（缺少 model_init_kwargs）提供基于 state_dict 的形状推断回退。
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn

from essence_forge.core.runtime_config import CFG
from essence_forge.core.models import EssenceForgeTCN, SimplifiedFftLwptSeTCN, TemporalConvNet


MODEL_CLASS_NAME_TEMPORAL = "temporal_convnet"
MODEL_CLASS_NAME_SIMPLIFIED = "simplified_fft_lwpt_se_tcn"
MODEL_CLASS_NAME_ESSENCE_FORGE = "essence_forge_tcn"

_MODEL_CLASS_REGISTRY: Dict[str, type[nn.Module]] = {
    MODEL_CLASS_NAME_TEMPORAL: TemporalConvNet,
    MODEL_CLASS_NAME_SIMPLIFIED: SimplifiedFftLwptSeTCN,
    MODEL_CLASS_NAME_ESSENCE_FORGE: EssenceForgeTCN,
}

_MODEL_CLASS_ALIASES: Dict[str, str] = {
    "tcn": MODEL_CLASS_NAME_TEMPORAL,
    "temporalconvnet": MODEL_CLASS_NAME_TEMPORAL,
    MODEL_CLASS_NAME_TEMPORAL: MODEL_CLASS_NAME_TEMPORAL,
    "fft_lwpt_se_tcn": MODEL_CLASS_NAME_SIMPLIFIED,
    "simplifiedfftlwptsetcn": MODEL_CLASS_NAME_SIMPLIFIED,
    MODEL_CLASS_NAME_SIMPLIFIED: MODEL_CLASS_NAME_SIMPLIFIED,
    "essence_forge": MODEL_CLASS_NAME_ESSENCE_FORGE,
    "essenceforgetcn": MODEL_CLASS_NAME_ESSENCE_FORGE,
    MODEL_CLASS_NAME_ESSENCE_FORGE: MODEL_CLASS_NAME_ESSENCE_FORGE,
}

_MODEL_INIT_KEYS_BY_CLASS: Dict[str, Tuple[str, ...]] = {
    name: tuple(
        key
        for key in inspect.signature(model_cls.__init__).parameters
        if key != "self"
    )
    for name, model_cls in _MODEL_CLASS_REGISTRY.items()
}


def normalize_model_class_name(model_class_name: str | None) -> str:
    if model_class_name is None:
        return MODEL_CLASS_NAME_TEMPORAL
    raw = str(model_class_name).strip().lower()
    if raw in _MODEL_CLASS_ALIASES:
        return _MODEL_CLASS_ALIASES[raw]
    supported = ", ".join(sorted(_MODEL_CLASS_REGISTRY.keys()))
    raise ValueError(f"未知 model_class_name={model_class_name}，可用值: {supported}")


def resolve_model_class(model_class_name: str | None) -> type[nn.Module]:
    normalized = normalize_model_class_name(model_class_name)
    return _MODEL_CLASS_REGISTRY[normalized]


def _model_init_keys(model_class_name: str | None) -> Tuple[str, ...]:
    normalized = normalize_model_class_name(model_class_name)
    return _MODEL_INIT_KEYS_BY_CLASS[normalized]


def _default_input_dim_from_cfg() -> int:
    base_raw_input_dim = int(getattr(CFG, "input_dim", 19))
    cross_sensor_residual_channels = (
        int(getattr(CFG, "cross_sensor_residual_channels", 9))
        if bool(getattr(CFG, "use_cross_sensor_residuals", False))
        else 0
    )
    raw_sensor_input_dim = base_raw_input_dim + cross_sensor_residual_channels
    mask_input_dim = (
        base_raw_input_dim if bool(getattr(CFG, "concat_health_mask_channels", False)) else 0
    )
    return raw_sensor_input_dim + mask_input_dim


def build_model_init_kwargs_from_cfg(
    overrides: Mapping[str, Any] | None = None,
    model_class_name: str | None = MODEL_CLASS_NAME_TEMPORAL,
) -> Dict[str, Any]:
    """
    基于当前 CFG 构建指定模型类的初始化参数。
    可用 overrides 覆盖部分字段（如 Optuna 输出）。
    """
    init_keys = _model_init_keys(model_class_name)
    kwargs: Dict[str, Any] = {}
    for key in init_keys:
        if key == "input_dim":
            kwargs[key] = _default_input_dim_from_cfg()
            continue
        if key == "num_classes":
            kwargs[key] = int(CFG.num_classes)
            continue
        if hasattr(CFG, key):
            kwargs[key] = getattr(CFG, key)

    if overrides:
        for key, value in overrides.items():
            if key in init_keys:
                kwargs[key] = value
    return kwargs


def _coerce_tuple_length(values: Iterable[int], target_len: int, fallback: int) -> Tuple[int, ...]:
    seq = tuple(int(v) for v in values)
    if target_len <= 0:
        return seq
    if len(seq) == target_len:
        return seq
    if len(seq) == 0:
        return tuple(int(fallback) for _ in range(target_len))
    if len(seq) > target_len:
        return seq[:target_len]
    # 长度不足时重复最后一个值补齐
    pad = tuple(int(seq[-1]) for _ in range(target_len - len(seq)))
    return seq + pad


def _has_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state_dict.keys())


def _infer_num_levels(state_dict: Mapping[str, torch.Tensor]) -> int:
    pattern = re.compile(r"^network\.(\d+)\.")
    levels = []
    for key in state_dict.keys():
        matched = pattern.match(key)
        if matched:
            levels.append(int(matched.group(1)))
    if not levels:
        return int(getattr(CFG, "tcn_num_levels", 4))
    return max(levels) + 1


def _infer_kernel_sizes_from_branches(
    state_dict: Mapping[str, torch.Tensor],
    branch_prefix: str,
) -> Tuple[int, ...]:
    pattern = re.compile(
        rf"^network\.0\.{re.escape(branch_prefix)}\.conv_branches\.(\d+)\.0\.conv\.weight$"
    )
    by_index: Dict[int, int] = {}
    for key, tensor in state_dict.items():
        matched = pattern.match(key)
        if matched and tensor.ndim == 3:
            branch_idx = int(matched.group(1))
            by_index[branch_idx] = int(tensor.shape[-1])
    if not by_index:
        return tuple()
    return tuple(kernel for _, kernel in sorted(by_index.items(), key=lambda item: item[0]))


def _infer_fused_kernel_sizes(state_dict: Mapping[str, torch.Tensor]) -> Tuple[int, ...]:
    # 取第 0 个 temporal block 的多尺度分支卷积核大小。
    return _infer_kernel_sizes_from_branches(
        state_dict,
        branch_prefix="multi_scale_conv",
    )


def _infer_hybrid_kernel_sizes(state_dict: Mapping[str, torch.Tensor]) -> Tuple[int, ...]:
    return _infer_kernel_sizes_from_branches(
        state_dict,
        branch_prefix="hybrid_multi_scale_conv",
    )


def _infer_lwpt_kernel_sizes(state_dict: Mapping[str, torch.Tensor]) -> Tuple[int, ...]:
    pattern = re.compile(r"^lwpt_frontend\.band_filters\.(\d+)\.weight$")
    by_index: Dict[int, int] = {}
    for key, tensor in state_dict.items():
        matched = pattern.match(key)
        if matched and tensor.ndim == 3:
            band_idx = int(matched.group(1))
            by_index[band_idx] = int(tensor.shape[-1])
    if not by_index:
        return tuple()
    return tuple(kernel for _, kernel in sorted(by_index.items(), key=lambda item: item[0]))


def _infer_tf2d_num_scales(state_dict: Mapping[str, torch.Tensor]) -> int:
    pattern = re.compile(r"^timefreq_encoder\.scale_encoders\.(\d+)\.")
    scales = []
    for key in state_dict.keys():
        matched = pattern.match(key)
        if matched:
            scales.append(int(matched.group(1)))
    if not scales:
        return 0
    return max(scales) + 1


def _infer_temporal_input_dim(state_dict: Mapping[str, torch.Tensor]) -> int | None:
    # 顺序：最稳定的首层输入权重。
    candidates = (
        "network.0.downsample.weight",
        "network.0.conv1.conv.weight",
        "network.0.conv1.weight",
        "network.0.multi_scale_conv.conv_branches.0.0.conv.weight",
    )
    for key in candidates:
        if key not in state_dict:
            continue
        tensor = state_dict[key]
        if tensor.ndim >= 2:
            return int(tensor.shape[1])
    return None


def _infer_simplified_input_dim(state_dict: Mapping[str, torch.Tensor]) -> int | None:
    # 优先从 LWPT depthwise 卷积反推输入通道（新旧结构都稳定）。
    lwpt_weight = state_dict.get("lwpt_frontend.band_filters.0.weight")
    if lwpt_weight is not None and lwpt_weight.ndim == 3:
        # depthwise Conv1d: [C, 1, K]
        return int(lwpt_weight.shape[0])

    tcn_in_channels: int | None = None
    candidates = (
        "network.0.downsample.weight",
        "network.0.hybrid_deform_conv.weight",
        "network.0.conv1.conv.weight",
        "network.0.conv1.weight",
    )
    for key in candidates:
        tensor = state_dict.get(key)
        if tensor is None or tensor.ndim < 2:
            continue
        tcn_in_channels = int(tensor.shape[1])
        break

    if tcn_in_channels is None:
        return None

    # 新结构存在独立频域分支，可直接用其输入通道确认 input_dim。
    freq_proj = state_dict.get("freq_projection.0.weight")
    if freq_proj is not None and freq_proj.ndim == 3:
        health_mask_proj = state_dict.get("health_mask_projection.weight")
        if health_mask_proj is not None and health_mask_proj.ndim == 3:
            return int(freq_proj.shape[1] + health_mask_proj.shape[1])
        return int(freq_proj.shape[1])

    # 旧结构为 early fusion，主干输入通道约等于 2 * input_dim。
    if tcn_in_channels % 2 == 0:
        return tcn_in_channels // 2
    return tcn_in_channels


def _infer_temporal_convnet_init_kwargs_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    对历史 TemporalConvNet checkpoint 进行形状关键参数推断。

    注意：
    - 某些不影响参数形状的超参（如若干 dropout/温度）无法从 state_dict 反推，
      会回退到当前 CFG。
    """
    kwargs = build_model_init_kwargs_from_cfg(
        model_class_name=MODEL_CLASS_NAME_TEMPORAL,
    )

    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is not None and classifier_weight.ndim == 2:
        kwargs["num_classes"] = int(classifier_weight.shape[0])
        kwargs["tcn_channels"] = int(classifier_weight.shape[1])

    inferred_input_dim = _infer_temporal_input_dim(state_dict)
    if inferred_input_dim is not None:
        kwargs["input_dim"] = inferred_input_dim

    kwargs["tcn_num_levels"] = _infer_num_levels(state_dict)

    conv2_weight = state_dict.get("network.0.conv2.conv.weight")
    if conv2_weight is None:
        conv2_weight = state_dict.get("network.0.conv2.weight")
    if conv2_weight is not None and conv2_weight.ndim == 3:
        kwargs["tcn_kernel_size"] = int(conv2_weight.shape[-1])

    # 前端开关
    kwargs["use_lwpt_frontend"] = _has_prefix(state_dict, "lwpt_frontend.band_filters.")
    if kwargs["use_lwpt_frontend"]:
        lwpt_kernel_sizes = _infer_lwpt_kernel_sizes(state_dict)
        if lwpt_kernel_sizes:
            kwargs["lwpt_kernel_sizes"] = lwpt_kernel_sizes
            kwargs["lwpt_num_bands"] = len(lwpt_kernel_sizes)

    # 主干与注意力
    kwargs["use_channel_se"] = _has_prefix(state_dict, "network.0.channel_se.")
    kwargs["use_fused_amtc_tcn"] = _has_prefix(state_dict, "network.0.multi_scale_conv.")
    if kwargs["use_fused_amtc_tcn"]:
        fused_kernel_sizes = _infer_fused_kernel_sizes(state_dict)
        if fused_kernel_sizes:
            kwargs["fused_tcn_kernel_sizes"] = fused_kernel_sizes
        kwargs["fused_tcn_use_pool_branch"] = _has_prefix(
            state_dict,
            "network.0.multi_scale_conv.pool_branch.",
        )

    # 时频分支
    kwargs["use_tf2d_branch"] = _has_prefix(state_dict, "timefreq_encoder.")
    if kwargs["use_tf2d_branch"]:
        tf_project = state_dict.get("timefreq_encoder.scale_encoders.0.project.0.weight")
        if tf_project is not None and tf_project.ndim == 4:
            kwargs["tf2d_branch_channels"] = int(tf_project.shape[0])

        num_scales = _infer_tf2d_num_scales(state_dict)
        if num_scales > 0:
            kwargs["tf2d_n_fft"] = _coerce_tuple_length(
                kwargs.get("tf2d_n_fft", (32,)),
                target_len=num_scales,
                fallback=32,
            )
            kwargs["tf2d_hop_lengths"] = _coerce_tuple_length(
                kwargs.get("tf2d_hop_lengths", (8,)),
                target_len=num_scales,
                fallback=8,
            )
            kwargs["tf2d_win_lengths"] = _coerce_tuple_length(
                kwargs.get("tf2d_win_lengths", (32,)),
                target_len=num_scales,
                fallback=32,
            )

        kwargs["use_layerwise_tf_fusion"] = _has_prefix(state_dict, "fusion_gates.")
        kwargs["use_global_branch_weighting"] = _has_prefix(state_dict, "branch_weight_mlp.")
    else:
        kwargs["use_layerwise_tf_fusion"] = False
        kwargs["use_global_branch_weighting"] = False

    return kwargs


def _infer_simplified_init_kwargs_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    kwargs = build_model_init_kwargs_from_cfg(
        model_class_name=MODEL_CLASS_NAME_SIMPLIFIED,
    )

    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is not None and classifier_weight.ndim == 2:
        kwargs["num_classes"] = int(classifier_weight.shape[0])
        kwargs["tcn_channels"] = int(classifier_weight.shape[1])

    inferred_input_dim = _infer_simplified_input_dim(state_dict)
    if inferred_input_dim is not None:
        kwargs["input_dim"] = inferred_input_dim

    kwargs["tcn_num_levels"] = _infer_num_levels(state_dict)

    conv2_weight = state_dict.get("network.0.conv2.conv.weight")
    if conv2_weight is None:
        conv2_weight = state_dict.get("network.0.conv2.weight")
    if conv2_weight is not None and conv2_weight.ndim == 3:
        kwargs["tcn_kernel_size"] = int(conv2_weight.shape[-1])

    kwargs["use_lwpt_frontend"] = _has_prefix(state_dict, "lwpt_frontend.band_filters.")
    if kwargs["use_lwpt_frontend"]:
        lwpt_kernel_sizes = _infer_lwpt_kernel_sizes(state_dict)
        if lwpt_kernel_sizes:
            kwargs["lwpt_kernel_sizes"] = lwpt_kernel_sizes
            kwargs["lwpt_num_bands"] = len(lwpt_kernel_sizes)

    kwargs["use_channel_se"] = _has_prefix(state_dict, "network.0.channel_se.")
    kwargs["use_sensor_group_attention"] = _has_prefix(
        state_dict,
        "sensor_group_attention.",
    )

    freq_proj = state_dict.get("freq_projection.0.weight")
    if freq_proj is not None and freq_proj.ndim == 3:
        kwargs["use_freq_branch"] = True
        kwargs["freq_branch_channels"] = int(freq_proj.shape[0])
        kwargs["freq_branch_kernel_size"] = int(freq_proj.shape[-1])
    else:
        kwargs["use_freq_branch"] = False
    kwargs["use_freq_branch_se"] = bool(kwargs["use_freq_branch"]) and _has_prefix(
        state_dict,
        "freq_channel_se.",
    )

    late_fusion_weight = state_dict.get("late_fusion_proj.0.weight")
    if late_fusion_weight is not None and late_fusion_weight.ndim == 2:
        time_channels = int(kwargs.get("tcn_channels", CFG.tcn_channels))
        freq_channels = (
            int(kwargs.get("freq_branch_channels", time_channels))
            if bool(kwargs.get("use_freq_branch", True))
            else 0
        )
        base_late_dim = time_channels + freq_channels
        late_in_features = int(late_fusion_weight.shape[1])
        if late_in_features == base_late_dim:
            kwargs["use_mixed_pooling"] = False
        elif late_in_features == base_late_dim * 2:
            kwargs["use_mixed_pooling"] = True

    conv1_has_deform = _has_prefix(state_dict, "network.0.conv1.offset_conv.") or _has_prefix(
        state_dict,
        "network.0.conv1.group_offset_convs.",
    )
    conv2_has_deform = _has_prefix(state_dict, "network.0.conv2.offset_conv.") or _has_prefix(
        state_dict,
        "network.0.conv2.group_offset_convs.",
    )
    hybrid_has_deform = _has_prefix(state_dict, "network.0.hybrid_deform_conv.offset_conv.") or _has_prefix(
        state_dict,
        "network.0.hybrid_deform_conv.group_offset_convs.",
    )
    kwargs["use_deformable_tcn"] = bool(conv1_has_deform or conv2_has_deform or hybrid_has_deform)
    kwargs["deformable_conv_mode"] = "both" if conv2_has_deform else "conv1_only"
    kwargs["deform_apply_levels"] = tuple(
        sorted(
            {
                int(key.split(".")[1])
                for key in state_dict.keys()
                if (
                    ".offset_conv." in key or ".group_offset_convs." in key
                )
                and key.startswith("network.")
            }
        )
    )
    kwargs["deform_group_mode"] = (
        "shared_by_group"
        if any(".group_offset_convs." in key for key in state_dict.keys())
        else "per_channel"
    )
    kwargs["deform_bypass_health_mask"] = _has_prefix(state_dict, "health_mask_projection.")
    kwargs["use_deform_conv_gate"] = _has_prefix(state_dict, "network.0.deform_conv_gate_logit")
    kwargs["use_hybrid_deform_multiscale_tcn"] = bool(
        _has_prefix(state_dict, "network.0.hybrid_deform_conv.")
        and _has_prefix(state_dict, "network.0.hybrid_multi_scale_conv.")
    )
    if kwargs["use_hybrid_deform_multiscale_tcn"]:
        hybrid_kernel_sizes = _infer_hybrid_kernel_sizes(state_dict)
        if hybrid_kernel_sizes:
            kwargs["hybrid_tcn_kernel_sizes"] = hybrid_kernel_sizes
    kwargs["use_levelwise_tcn_aggregation"] = _has_prefix(state_dict, "levelwise_score.")
    kwargs["use_uncertainty_temporal_pooling"] = _has_prefix(state_dict, "uncertainty_gate.")
    if kwargs["use_deformable_tcn"]:
        kwargs["deformable_causal_mode"] = str(getattr(CFG, "deformable_causal_mode", "strict"))
        offset_weight = state_dict.get("network.0.conv1.offset_conv.weight")
        if offset_weight is None:
            offset_weight = state_dict.get("network.0.conv2.offset_conv.weight")
        if offset_weight is None:
            offset_weight = state_dict.get("network.0.hybrid_deform_conv.offset_conv.weight")
        if offset_weight is None:
            offset_weight = state_dict.get("network.0.conv1.group_offset_convs.0.weight")
        if offset_weight is None:
            offset_weight = state_dict.get("network.0.conv2.group_offset_convs.0.weight")
        if offset_weight is None:
            offset_weight = state_dict.get("network.0.hybrid_deform_conv.group_offset_convs.0.weight")
        if offset_weight is not None and offset_weight.ndim == 3:
            kwargs["deform_offset_kernel_size"] = int(offset_weight.shape[-1])

    return kwargs


def infer_model_init_kwargs_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    model_class_name: str | None = MODEL_CLASS_NAME_TEMPORAL,
) -> Dict[str, Any]:
    normalized = normalize_model_class_name(model_class_name)
    if normalized == MODEL_CLASS_NAME_TEMPORAL:
        return _infer_temporal_convnet_init_kwargs_from_state_dict(state_dict)
    if normalized in {MODEL_CLASS_NAME_SIMPLIFIED, MODEL_CLASS_NAME_ESSENCE_FORGE}:
        return _infer_simplified_init_kwargs_from_state_dict(state_dict)
    raise ValueError(f"未实现该模型的 state_dict 推断: {normalized}")


def _load_checkpoint_payload(
    checkpoint: str | Path | Mapping[str, Any],
    map_location: torch.device,
) -> Dict[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(checkpoint, map_location=map_location)
    elif isinstance(checkpoint, Mapping):
        payload = dict(checkpoint)
    else:
        raise TypeError(
            "checkpoint 必须是路径(str/Path)或包含 state_dict 的映射对象"
        )

    if "state_dict" not in payload:
        raise KeyError("checkpoint 缺少 'state_dict' 字段")
    if not isinstance(payload["state_dict"], Mapping):
        raise TypeError("checkpoint['state_dict'] 必须是映射类型")
    return payload


def _freeze_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_for_hash(item))
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_hash(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_for_hash(item) for item in value))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def load_temporal_convnet_from_checkpoint(
    checkpoint: str | Path | Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    加载模型 checkpoint，并返回:
    1. 已加载权重的模型
    2. checkpoint payload
    3. 元信息（包含使用的 model_init_kwargs / model_class_name 及来源）
    """
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    payload = _load_checkpoint_payload(checkpoint=checkpoint, map_location=resolved_device)
    state_dict = payload["state_dict"]

    resolved_model_class_name = normalize_model_class_name(payload.get("model_class_name"))
    model_class = resolve_model_class(resolved_model_class_name)

    attempts: list[tuple[str, Dict[str, Any]]] = []

    saved_kwargs = payload.get("model_init_kwargs")
    if isinstance(saved_kwargs, Mapping):
        merged = build_model_init_kwargs_from_cfg(
            overrides=saved_kwargs,
            model_class_name=resolved_model_class_name,
        )
        attempts.append(("checkpoint", merged))

    attempts.append(
        (
            "default_cfg",
            build_model_init_kwargs_from_cfg(model_class_name=resolved_model_class_name),
        )
    )
    attempts.append(
        (
            "inferred_from_state_dict",
            infer_model_init_kwargs_from_state_dict(
                state_dict,
                model_class_name=resolved_model_class_name,
            ),
        )
    )

    seen = set()
    deduped_attempts: list[tuple[str, Dict[str, Any]]] = []
    for source, kwargs in attempts:
        key = (
            source,
            tuple(
                (item_key, _freeze_for_hash(item_value))
                for item_key, item_value in sorted(kwargs.items(), key=lambda item: item[0])
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_attempts.append((source, kwargs))

    errors: list[tuple[str, str]] = []
    for source, model_kwargs in deduped_attempts:
        try:
            model = model_class(**model_kwargs).to(resolved_device)
            model.load_state_dict(state_dict)
            return model, payload, {
                "model_kwargs_source": source,
                "model_init_kwargs": model_kwargs,
                "model_class_name": resolved_model_class_name,
            }
        except RuntimeError as exc:
            errors.append((source, str(exc)))

    detail = "\n".join([f"- {source}: {message}" for source, message in errors])
    raise RuntimeError(
        "无法加载模型 checkpoint，所有架构匹配尝试均失败。\n"
        f"model_class_name={resolved_model_class_name}\n"
        f"已尝试来源:\n{detail}"
    )
