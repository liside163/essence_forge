"""
uav_tl.models 包

模型模块：TCN 基线与 LWPT/TF2D 前端。
"""

from essence_forge.core.models.lwpt import LearnableWaveletPacketFrontend
from essence_forge.core.models.masking import masked_max_pooling, masked_mean_pooling
from essence_forge.core.models.sensor_group_attention import SensorGroupAttention
from essence_forge.core.models.simplified_fft_lwpt_se_tcn import SimplifiedFftLwptSeTCN
from essence_forge.core.models.tcn import TemporalConvNet
from essence_forge.core.models.timefreq import (
    LayerwiseFusionGate,
    MultiResolutionSTFT,
    ResMultiScaleConv2D,
    TimeFreqAlign1D,
    TimeFreqEncoder2D,
)

__all__ = [
    "EssenceForgeTCN",
    "LearnableWaveletPacketFrontend",
    "LayerwiseFusionGate",
    "MultiResolutionSTFT",
    "ResMultiScaleConv2D",
    "SensorGroupAttention",
    "SimplifiedFftLwptSeTCN",
    "TemporalConvNet",
    "TimeFreqAlign1D",
    "TimeFreqEncoder2D",
    "masked_max_pooling",
    "masked_mean_pooling",
]


def __getattr__(name: str):
    if name == "EssenceForgeTCN":
        from essence_forge.model import EssenceForgeTCN

        return EssenceForgeTCN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
