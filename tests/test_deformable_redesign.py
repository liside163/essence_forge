from __future__ import annotations

import json
from pathlib import Path

import torch

from essence_forge.core.channel_layout import DEFAULT_30_RAW_CHANNEL_NAMES
from essence_forge.core.model_checkpoint import load_temporal_convnet_from_checkpoint
from essence_forge.core.models.tcn import CausalConv1d, DeformableCausalConv1d
from essence_forge.core.runtime_config import CFG, load_config
from essence_forge.model import EssenceForgeTCN


def _set_raw30_cfg() -> None:
    CFG.values["input_dim"] = 30
    CFG.values["channels"] = DEFAULT_30_RAW_CHANNEL_NAMES
    CFG.values["use_cross_sensor_residuals"] = False
    CFG.values["cross_sensor_residual_channels"] = 0
    CFG.values["cross_sensor_residual_channel_names"] = ()
    CFG.values["concat_health_mask_channels"] = True


def _build_redesign_model(**overrides) -> EssenceForgeTCN:
    kwargs = dict(
        input_dim=60,
        num_classes=4,
        tcn_channels=12,
        tcn_num_levels=3,
        tcn_kernel_size=3,
        tcn_dropout=0.1,
        use_lwpt_frontend=False,
        lwpt_num_bands=4,
        lwpt_kernel_sizes=(3, 5, 7, 9),
        lwpt_dropout=0.1,
        use_channel_se=False,
        channel_se_reduction=4,
        use_mixed_pooling=False,
        use_freq_branch=True,
        freq_branch_channels=10,
        freq_branch_kernel_size=3,
        freq_branch_dropout=0.1,
        use_freq_branch_se=False,
        freq_branch_se_reduction=4,
        concat_health_mask_channels=True,
        use_sensor_group_attention=False,
        fft_mag_log1p=True,
        fft_mag_norm="none",
        use_deformable_tcn=True,
        deformable_conv_mode="conv1_only",
        deformable_causal_mode="strict",
        deform_offset_kernel_size=3,
        deform_offset_l1_weight=5e-4,
        use_hybrid_deform_multiscale_tcn=False,
        hybrid_tcn_kernel_sizes=(3, 5, 7),
        use_levelwise_tcn_aggregation=False,
        use_uncertainty_temporal_pooling=False,
        deform_apply_levels=(0,),
        deform_scope="raw_only",
        deform_group_mode="shared_by_group",
        deform_groups=(
            "accel",
            "gyro",
            "mag",
            "pos",
            "vel",
            "quat",
            "actuator_rpm",
            "baro",
        ),
        deform_max_offset_scale=1.0,
        deform_zero_init=True,
        deform_warmup_epochs=5,
        deform_offset_lr_scale=0.1,
        deform_offset_tv_weight=1e-3,
        deform_bypass_health_mask=True,
    )
    kwargs.update(overrides)
    return EssenceForgeTCN(**kwargs)


def test_health_mask_bypass_reduces_feature_branch_input_to_raw_channels(monkeypatch) -> None:
    _set_raw30_cfg()
    model = _build_redesign_model()

    assert model.raw_sensor_input_dim == 30
    assert model.health_mask_input_dim == 30
    assert model.deform_bypass_health_mask is True
    assert model.time_input_channels == 30
    assert model.freq_input_channels == 30
    assert model.health_mask_projection is not None

    x = torch.randn(2, 12, 60)
    lengths = torch.tensor([12, 9], dtype=torch.long)
    logits, aux = model.forward_with_aux(x, lengths)

    assert logits.shape == (2, 4)
    assert aux["health_mask_bypass_enabled"] is True
    assert aux["feature_branch_input_shape"] == (2, 30, 12)
    assert aux["health_mask_shape"] == (2, 30, 12)
    assert aux["fft_shape"] == (2, 30, 12)


def test_grouped_zero_init_deformable_conv_matches_standard_causal_conv() -> None:
    torch.manual_seed(7)
    standard = CausalConv1d(in_channels=4, out_channels=5, kernel_size=3, dilation=2)
    deform = DeformableCausalConv1d(
        in_channels=4,
        out_channels=5,
        kernel_size=3,
        dilation=2,
        offset_kernel_size=3,
        causal_mode="strict",
        group_mode="shared_by_group",
        channel_groups=((0, 1), (2, 3)),
        max_offset_scale=1.0,
        zero_init=True,
    )
    with torch.no_grad():
        deform.weight.copy_(standard.conv.weight)
        deform.bias.copy_(standard.conv.bias)

    x = torch.randn(3, 4, 16)
    y_standard = standard(x)
    y_deform = deform(x)

    assert torch.allclose(y_deform, y_standard, atol=1e-6)


def test_grouped_deformable_offsets_are_shared_and_bounded() -> None:
    torch.manual_seed(11)
    deform = DeformableCausalConv1d(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        dilation=2,
        offset_kernel_size=3,
        causal_mode="strict",
        group_mode="shared_by_group",
        channel_groups=((0, 1), (2, 3)),
        max_offset_scale=1.0,
        zero_init=False,
    )
    x = torch.randn(2, 4, 10)
    offsets = deform._build_offsets(x)

    assert offsets.shape == (2, 4, 3, 10)
    assert torch.all(offsets <= 0.0)
    assert torch.all(offsets >= -2.0)
    assert torch.allclose(offsets[:, 0], offsets[:, 1], atol=1e-6)
    assert torch.allclose(offsets[:, 2], offsets[:, 3], atol=1e-6)


def test_grouped_offset_build_matches_group_branch_dtype() -> None:
    class HalfOffsetConv(torch.nn.Module):
        def __init__(self, kernel_size: int) -> None:
            super().__init__()
            self.kernel_size = kernel_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (x.shape[0], self.kernel_size, x.shape[-1] - 2),
                dtype=torch.float16,
                device=x.device,
            )

    deform = DeformableCausalConv1d(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        dilation=2,
        offset_kernel_size=3,
        causal_mode="strict",
        group_mode="shared_by_group",
        channel_groups=((0, 1), (2, 3)),
        max_offset_scale=1.0,
        zero_init=False,
    )
    deform.group_offset_convs = torch.nn.ModuleList([HalfOffsetConv(3), HalfOffsetConv(3)])

    offsets = deform._build_offsets(torch.randn(2, 4, 10, dtype=torch.float32))

    assert offsets.dtype == torch.float16


def test_model_applies_deform_only_to_selected_levels() -> None:
    _set_raw30_cfg()
    model = _build_redesign_model(
        input_dim=30,
        concat_health_mask_channels=False,
        deform_bypass_health_mask=False,
        deform_apply_levels=(0,),
    )

    assert isinstance(model.network[0].conv1, DeformableCausalConv1d)
    assert not isinstance(model.network[1].conv1, DeformableCausalConv1d)
    assert not isinstance(model.network[2].conv1, DeformableCausalConv1d)


def test_optional_block_level_deform_gate_defaults_toward_standard_conv() -> None:
    _set_raw30_cfg()
    model = _build_redesign_model(
        input_dim=30,
        concat_health_mask_channels=False,
        deform_bypass_health_mask=False,
        use_deform_conv_gate=True,
        deform_conv_gate_bias_init=-2.0,
    )

    gate = model.network[0].deform_conv_gate_logit

    assert gate is not None
    assert torch.sigmoid(gate.detach()).item() < 0.5


def test_runtime_config_and_checkpoint_round_trip_new_deform_options(tmp_path: Path) -> None:
    config_path = Path("configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["cross_sensor_residuals"]["enable"] = False
    payload["model"]["use_deformable_tcn"] = True
    payload["model"]["deform_apply_levels"] = [0]
    payload["model"]["deform_scope"] = "raw_only"
    payload["model"]["deform_group_mode"] = "shared_by_group"
    payload["model"]["deform_groups"] = [
        "accel",
        "gyro",
        "mag",
        "pos",
        "vel",
        "quat",
        "actuator_rpm",
        "baro",
    ]
    payload["model"]["deform_max_offset_scale"] = 1.0
    payload["model"]["deform_zero_init"] = True
    payload["model"]["deform_warmup_epochs"] = 5
    payload["model"]["deform_offset_lr_scale"] = 0.1
    payload["model"]["deform_offset_tv_weight"] = 1e-3
    payload["model"]["deform_bypass_health_mask"] = True
    payload["model"]["use_deform_conv_gate"] = True
    payload["model"]["deform_conv_gate_bias_init"] = -2.0

    custom_config = tmp_path / "config.json"
    custom_config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_config(custom_config)
    assert loaded.deform_apply_levels == (0,)
    assert loaded.deform_scope == "raw_only"
    assert loaded.deform_group_mode == "shared_by_group"
    assert loaded.deform_groups == (
        "accel",
        "gyro",
        "mag",
        "pos",
        "vel",
        "quat",
        "actuator_rpm",
        "baro",
    )
    assert loaded.deform_max_offset_scale == 1.0
    assert loaded.deform_zero_init is True
    assert loaded.deform_warmup_epochs == 5
    assert loaded.deform_offset_lr_scale == 0.1
    assert loaded.deform_offset_tv_weight == 1e-3
    assert loaded.deform_bypass_health_mask is True
    assert loaded.use_deform_conv_gate is True
    assert loaded.deform_conv_gate_bias_init == -2.0

    _set_raw30_cfg()
    model = _build_redesign_model(
        use_deform_conv_gate=True,
        deform_conv_gate_bias_init=-2.0,
    )
    checkpoint_path = tmp_path / "redesign.pt"
    torch.save(
        {
            "model_name": "redesign",
            "model_class_name": "essence_forge_tcn",
            "epoch": 0,
            "state_dict": model.state_dict(),
            "model_init_kwargs": {
                "input_dim": 60,
                "num_classes": 4,
                "tcn_channels": 12,
                "tcn_num_levels": 3,
                "tcn_kernel_size": 3,
                "tcn_dropout": 0.1,
                "use_lwpt_frontend": False,
                "lwpt_num_bands": 4,
                "lwpt_kernel_sizes": (3, 5, 7, 9),
                "lwpt_dropout": 0.1,
                "use_channel_se": False,
                "channel_se_reduction": 4,
                "use_mixed_pooling": False,
                "use_freq_branch": True,
                "freq_branch_channels": 10,
                "freq_branch_kernel_size": 3,
                "freq_branch_dropout": 0.1,
                "use_freq_branch_se": False,
                "freq_branch_se_reduction": 4,
                "concat_health_mask_channels": True,
                "use_sensor_group_attention": False,
                "fft_mag_log1p": True,
                "fft_mag_norm": "none",
                "use_deformable_tcn": True,
                "deformable_conv_mode": "conv1_only",
                "deformable_causal_mode": "strict",
                "deform_offset_kernel_size": 3,
                "deform_offset_l1_weight": 5e-4,
                "use_hybrid_deform_multiscale_tcn": False,
                "hybrid_tcn_kernel_sizes": (3, 5, 7),
                "use_levelwise_tcn_aggregation": False,
                "use_uncertainty_temporal_pooling": False,
                "deform_apply_levels": (0,),
                "deform_scope": "raw_only",
                "deform_group_mode": "shared_by_group",
                "deform_groups": (
                    "accel",
                    "gyro",
                    "mag",
                    "pos",
                    "vel",
                    "quat",
                    "actuator_rpm",
                    "baro",
                ),
                "deform_max_offset_scale": 1.0,
                "deform_zero_init": True,
                "deform_warmup_epochs": 5,
                "deform_offset_lr_scale": 0.1,
                "deform_offset_tv_weight": 1e-3,
                "deform_bypass_health_mask": True,
                "use_deform_conv_gate": True,
                "deform_conv_gate_bias_init": -2.0,
            },
        },
        checkpoint_path,
    )

    loaded_model, _, meta = load_temporal_convnet_from_checkpoint(checkpoint_path)

    assert isinstance(loaded_model, EssenceForgeTCN)
    assert meta["model_init_kwargs"]["deform_apply_levels"] == (0,)
    assert meta["model_init_kwargs"]["deform_scope"] == "raw_only"
    assert meta["model_init_kwargs"]["deform_group_mode"] == "shared_by_group"
    assert meta["model_init_kwargs"]["deform_bypass_health_mask"] is True
    assert meta["model_init_kwargs"]["use_deform_conv_gate"] is True
