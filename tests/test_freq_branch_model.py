from __future__ import annotations

import torch

from essence_forge.core.model_checkpoint import load_temporal_convnet_from_checkpoint
from essence_forge.model import EssenceForgeTCN


def _build_small_model(*, use_freq_branch: bool) -> EssenceForgeTCN:
    return EssenceForgeTCN(
        input_dim=6,
        num_classes=3,
        tcn_channels=8,
        tcn_num_levels=2,
        tcn_kernel_size=3,
        tcn_dropout=0.1,
        use_lwpt_frontend=False,
        lwpt_num_bands=4,
        lwpt_kernel_sizes=(3, 5, 7, 9),
        lwpt_dropout=0.1,
        use_channel_se=False,
        channel_se_reduction=4,
        use_mixed_pooling=False,
        use_freq_branch=use_freq_branch,
        freq_branch_channels=8,
        freq_branch_kernel_size=3,
        freq_branch_dropout=0.1,
        use_freq_branch_se=False,
        freq_branch_se_reduction=4,
        concat_health_mask_channels=False,
        use_sensor_group_attention=False,
        fft_mag_log1p=True,
        fft_mag_norm="none",
        use_deformable_tcn=False,
        deformable_conv_mode="conv1_only",
        deformable_causal_mode="strict",
        deform_offset_kernel_size=3,
        deform_offset_l1_weight=0.0,
        use_hybrid_deform_multiscale_tcn=False,
        hybrid_tcn_kernel_sizes=(3, 5, 7),
        use_levelwise_tcn_aggregation=False,
        use_uncertainty_temporal_pooling=False,
    )


def test_model_forward_without_frequency_branch() -> None:
    model = _build_small_model(use_freq_branch=False)

    x = torch.randn(2, 16, 6)
    lengths = torch.tensor([16, 13], dtype=torch.long)

    logits, aux = model.forward_with_aux(x, lengths)

    assert logits.shape == (2, 3)
    assert aux["freq_stream_shape"] is None
    assert aux["freq_pooled_shape"] is None
    assert aux["late_fusion_shape"] == (2, 8)


def test_freeze_frontend_skips_frequency_projection_when_branch_disabled() -> None:
    model = _build_small_model(use_freq_branch=False)

    model.freeze_frontend(True)

    assert all(param.requires_grad is False for param in model.lwpt_frontend.parameters())
    assert model.freq_projection is None


def test_checkpoint_round_trip_without_frequency_branch(tmp_path) -> None:
    model = _build_small_model(use_freq_branch=False)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_name": "freqless",
            "model_class_name": "essence_forge_tcn",
            "epoch": 0,
            "state_dict": model.state_dict(),
            "model_init_kwargs": {
                "input_dim": 6,
                "num_classes": 3,
                "tcn_channels": 8,
                "tcn_num_levels": 2,
                "tcn_kernel_size": 3,
                "tcn_dropout": 0.1,
                "use_lwpt_frontend": False,
                "use_channel_se": False,
                "use_mixed_pooling": False,
                "use_freq_branch": False,
                "freq_branch_channels": 8,
                "freq_branch_kernel_size": 3,
                "freq_branch_dropout": 0.1,
                "use_freq_branch_se": False,
                "concat_health_mask_channels": False,
                "use_sensor_group_attention": False,
                "fft_mag_log1p": True,
                "fft_mag_norm": "none",
                "use_deformable_tcn": False,
                "deformable_conv_mode": "conv1_only",
                "deformable_causal_mode": "strict",
                "deform_offset_kernel_size": 3,
                "deform_offset_l1_weight": 0.0,
                "use_hybrid_deform_multiscale_tcn": False,
                "hybrid_tcn_kernel_sizes": (3, 5, 7),
                "use_levelwise_tcn_aggregation": False,
                "use_uncertainty_temporal_pooling": False,
            },
        },
        checkpoint_path,
    )

    loaded_model, _, meta = load_temporal_convnet_from_checkpoint(checkpoint_path)

    assert isinstance(loaded_model, EssenceForgeTCN)
    assert meta["model_init_kwargs"]["use_freq_branch"] is False
    assert loaded_model.use_freq_branch is False
    assert loaded_model.freq_projection is None
