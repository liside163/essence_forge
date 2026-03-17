"""
tcn.py

TCN主体网络（精简版）：
1. 支持 LWPT 可学习频带前端。
2. 支持将 AMTC 多尺度能力融合到 TCN 卷积块（Fused-AMTC-TCN）。
3. 支持 "1D时域 + 2D时频" 双分支分层门控融合。
4. 支持 ChannelSE 通道注意力。

精简记录（2026-02-20）：
- 删除 LeadFollowerAttention (LFSGA)：实验证明降低源域 F1 约14%
- 删除 PhysicsResidualModule：实验证明降低源域 F1 约5%
- 删除 AMCFBank 前端：贡献仅 0.025 F1，噪声级
- 删除 AMTCFrontend：最差表现 F1=0.12，已淘汰
- 删除 PrototypeMetricHead：无消融证据
- 删除 SelfAttentionPooling1D：Optuna 未选中
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from essence_forge.core.runtime_config import CFG
from essence_forge.core.models.lwpt import LearnableWaveletPacketFrontend
from essence_forge.core.models.masking import masked_mean_pooling
from essence_forge.core.models.timefreq import (
    LayerwiseFusionGate,
    MultiResolutionSTFT,
    TimeFreqAlign1D,
    TimeFreqEncoder2D,
    TimeFreqSpecAugmentor,
)


def _to_kernel_tuple(kernel_sizes: Iterable[int], name: str) -> Tuple[int, ...]:
    """
    标准化卷积核参数并做合法性校验。
    为什么要做这个步骤：
    1. 配置通常从JSON list传入，统一为tuple便于内部稳定使用。
    2. 多尺度卷积分支需要提前校验，避免运行期才触发错误。
    3. 这里要求正奇数卷积核，便于保持长度不变并保持对称感受野。
    """
    kernels = tuple(int(k) for k in kernel_sizes)
    if len(kernels) == 0:
        raise ValueError(f"{name} 不能为空")
    for kernel in kernels:
        if kernel <= 0:
            raise ValueError(f"{name} 中卷积核必须是正整数，当前={kernel}")
        if kernel % 2 == 0:
            raise ValueError(f"{name} 中卷积核必须是奇数，当前={kernel}")
    return kernels


# ===========================================================================
#  基础卷积/池化组件
# ===========================================================================

class CausalConv1d(nn.Module):
    """
    因果1D卷积（只看过去，不看未来）。

    维度变化：
    输入: [B, C_in, T]
    输出: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.groups = int(groups)
        if self.groups <= 0:
            raise ValueError("groups 必须 > 0")
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            groups=self.groups,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播。
        输入: [B, C_in, T]
        输出: [B, C_out, T]
        """
        # 因果填充：只在左侧填充，保证不使用未来信息
        pad_left = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (pad_left, 0))
        return self.conv(x_padded)


class DeformableCausalConv1d(nn.Module):
    """
    1D 可变形因果卷积（纯 PyTorch 实现）。

    关键点：
    1. 先用 depthwise 因果卷积回归 offset，输出 [B, C_in, K, T]。
    2. 使用 floor/ceil + 线性插值做可微采样（gather 实现）。
    3. 使用 einsum 执行分组卷积聚合，输出 [B, C_out, T]。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        offset_kernel_size: int = 3,
        causal_mode: str = "strict",
        groups: int = 1,
        bias: bool = True,
        group_mode: str = "per_channel",
        channel_groups: Sequence[Sequence[int]] | None = None,
        max_offset_scale: float = 0.0,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels 必须 > 0")
        if out_channels <= 0:
            raise ValueError("out_channels 必须 > 0")
        if kernel_size <= 0:
            raise ValueError("kernel_size 必须 > 0")
        if dilation <= 0:
            raise ValueError("dilation 必须 > 0")
        if offset_kernel_size <= 1 or offset_kernel_size % 2 == 0:
            raise ValueError("offset_kernel_size 必须是大于1的奇数")
        if groups <= 0:
            raise ValueError("groups 必须 > 0")
        if in_channels % groups != 0:
            raise ValueError("in_channels 必须能被 groups 整除")
        if out_channels % groups != 0:
            raise ValueError("out_channels 必须能被 groups 整除")

        causal_mode_normalized = str(causal_mode).strip().lower()
        if causal_mode_normalized not in {"strict", "relaxed"}:
            raise ValueError("causal_mode 必须是 strict 或 relaxed")
        group_mode_normalized = str(group_mode).strip().lower()
        if group_mode_normalized not in {"per_channel", "shared_by_group"}:
            raise ValueError("group_mode 必须是 per_channel 或 shared_by_group")
        if float(max_offset_scale) < 0.0:
            raise ValueError("max_offset_scale 必须 >= 0")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.offset_kernel_size = int(offset_kernel_size)
        self.causal_mode = causal_mode_normalized
        self.groups = int(groups)
        self.group_mode = group_mode_normalized
        self.max_offset_scale = float(max_offset_scale)
        self.zero_init = bool(zero_init)
        self.channel_groups = self._normalize_channel_groups(
            channel_groups=channel_groups,
            in_channels=self.in_channels,
            group_mode=self.group_mode,
        )

        # 参数形状与 Conv1d 保持一致: [C_out, C_in/groups, K]
        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels // self.groups, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bool(bias) else None
        self.reset_parameters()

        # per-channel K offsets: depthwise causal conv outputs [B, C*K, T]
        self.offset_conv: nn.Conv1d | None = None
        self.group_offset_convs: nn.ModuleList | None = None
        if self.group_mode == "per_channel":
            self.offset_conv = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.in_channels * self.kernel_size,
                kernel_size=self.offset_kernel_size,
                stride=1,
                padding=0,
                groups=self.in_channels,
                bias=True,
            )
        else:
            self.group_offset_convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=len(group),
                        out_channels=self.kernel_size,
                        kernel_size=self.offset_kernel_size,
                        stride=1,
                        padding=0,
                        groups=1,
                        bias=True,
                    )
                    for group in self.channel_groups
                ]
            )
        self._reset_offset_parameters()
        self._last_offset_l1: torch.Tensor | None = None
        self._last_offset_tv: torch.Tensor | None = None
        self._last_offset_saturation_ratio: torch.Tensor | None = None

    @staticmethod
    def _normalize_channel_groups(
        *,
        channel_groups: Sequence[Sequence[int]] | None,
        in_channels: int,
        group_mode: str,
    ) -> tuple[tuple[int, ...], ...]:
        if group_mode == "per_channel":
            return tuple((idx,) for idx in range(int(in_channels)))
        if channel_groups is None or len(channel_groups) == 0:
            raise ValueError("shared_by_group 模式必须显式提供 channel_groups")
        normalized = tuple(tuple(int(idx) for idx in group) for group in channel_groups)
        flat = [idx for group in normalized for idx in group]
        if len(flat) != int(in_channels):
            raise ValueError(
                f"channel_groups 必须完整覆盖所有输入通道，当前覆盖={len(flat)} 期望={in_channels}"
            )
        if len(set(flat)) != len(flat):
            raise ValueError("channel_groups 含重复通道索引")
        if min(flat) < 0 or max(flat) >= int(in_channels):
            raise ValueError("channel_groups 含越界通道索引")
        return normalized

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = (self.in_channels // self.groups) * self.kernel_size
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _reset_offset_parameters(self) -> None:
        init_bias = -20.0 if self.max_offset_scale > 0.0 and self.causal_mode == "strict" else 0.0
        if self.offset_conv is not None:
            if self.zero_init:
                nn.init.zeros_(self.offset_conv.weight)
                if self.offset_conv.bias is not None:
                    nn.init.constant_(self.offset_conv.bias, init_bias)
            return
        if self.group_offset_convs is None:
            return
        for conv in self.group_offset_convs:
            if self.zero_init:
                nn.init.zeros_(conv.weight)
                if conv.bias is not None:
                    nn.init.constant_(conv.bias, init_bias)

    def get_last_offset_l1(self) -> torch.Tensor:
        if self._last_offset_l1 is None:
            return self.weight.new_zeros(())
        return self._last_offset_l1

    def get_last_offset_tv(self) -> torch.Tensor:
        if self._last_offset_tv is None:
            return self.weight.new_zeros(())
        return self._last_offset_tv

    def get_last_offset_saturation_ratio(self) -> torch.Tensor:
        if self._last_offset_saturation_ratio is None:
            return self.weight.new_zeros(())
        return self._last_offset_saturation_ratio

    def iter_offset_parameters(self):
        if self.offset_conv is not None:
            yield from self.offset_conv.parameters()
        if self.group_offset_convs is not None:
            for conv in self.group_offset_convs:
                yield from conv.parameters()

    def _parameterize_offsets(self, offset_raw: torch.Tensor) -> torch.Tensor:
        if self.max_offset_scale > 0.0:
            max_offset = float(self.dilation) * float(self.max_offset_scale)
            if self.causal_mode == "strict":
                return -max_offset * torch.sigmoid(offset_raw)
            return max_offset * torch.tanh(offset_raw)
        if self.causal_mode == "strict":
            return -offset_raw.abs()
        return offset_raw

    def _build_offsets(self, x: torch.Tensor) -> torch.Tensor:
        pad_left = self.offset_kernel_size - 1
        x_padded = F.pad(x, (pad_left, 0))
        if self.offset_conv is not None:
            offset_raw = self.offset_conv(x_padded)  # [B, C*K, T]
            bsz, _, seq_len = offset_raw.shape
            offsets = offset_raw.view(bsz, self.in_channels, self.kernel_size, seq_len)
            return self._parameterize_offsets(offsets)

        if self.group_offset_convs is None:
            raise RuntimeError("offset branch 未初始化")

        bsz, _, seq_len = x.shape
        offsets: torch.Tensor | None = None
        for group_indices, group_conv in zip(self.channel_groups, self.group_offset_convs):
            group_x = x_padded[:, group_indices, :]
            group_raw = group_conv(group_x)  # [B, K, T]
            group_offsets = self._parameterize_offsets(group_raw).unsqueeze(1)
            if offsets is None:
                offsets = group_offsets.new_zeros((bsz, self.in_channels, self.kernel_size, seq_len))
            offsets[:, group_indices, :, :] = group_offsets.expand(
                -1, len(group_indices), -1, -1
            )
        if offsets is None:
            raise RuntimeError("shared_by_group 模式下未生成任何 offset")
        return offsets

    def _deform_sample(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        bsz, channels, seq_len = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"输入通道不匹配，期望={self.in_channels}，实际={channels}"
            )

        dtype = x.dtype
        device = x.device
        pad_left = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (pad_left, 0))
        padded_seq_len = x_padded.shape[-1]
        base_t = (
            torch.arange(seq_len, device=device, dtype=dtype).view(1, 1, 1, seq_len)
            + float(pad_left)
        )
        kernel_offsets = torch.arange(self.kernel_size, device=device, dtype=dtype).view(
            1, 1, self.kernel_size, 1
        )
        reverse_offsets = float(self.kernel_size - 1) - kernel_offsets
        base_positions = base_t - reverse_offsets * float(self.dilation)
        positions = base_positions + offsets
        rounded_positions = positions.round()
        positions = torch.where(
            (positions - rounded_positions).abs() < 1e-6,
            rounded_positions,
            positions,
        )
        # Guard against NaN/Inf offsets: invalid gather indices on CUDA would
        # trigger device-side assert and abort the whole trial process.
        positions = torch.nan_to_num(
            positions,
            nan=0.0,
            posinf=float(padded_seq_len - 1),
            neginf=0.0,
        ).clamp(min=0.0, max=float(padded_seq_len - 1))

        idx0 = torch.floor(positions)
        idx1 = torch.clamp(idx0 + 1.0, max=float(padded_seq_len - 1))
        alpha = torch.nan_to_num(
            positions - idx0,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(min=0.0, max=1.0)

        x_expanded = x_padded.unsqueeze(2).expand(-1, -1, self.kernel_size, -1)
        idx0_long = idx0.to(dtype=torch.long).clamp(min=0, max=padded_seq_len - 1)
        idx1_long = idx1.to(dtype=torch.long).clamp(min=0, max=padded_seq_len - 1)
        sample0 = torch.gather(x_expanded, dim=3, index=idx0_long)
        sample1 = torch.gather(x_expanded, dim=3, index=idx1_long)
        sampled = sample0 * (1.0 - alpha) + sample1 * alpha
        return sampled

    def _grouped_einsum_conv(self, sampled: torch.Tensor) -> torch.Tensor:
        bsz, _, _, seq_len = sampled.shape
        if self.groups == 1:
            y = torch.einsum("ock,bckt->bot", self.weight, sampled)
        else:
            channels_per_group = self.in_channels // self.groups
            out_per_group = self.out_channels // self.groups
            sampled_grouped = sampled.view(
                bsz,
                self.groups,
                channels_per_group,
                self.kernel_size,
                seq_len,
            )
            weight_grouped = self.weight.view(
                self.groups,
                out_per_group,
                channels_per_group,
                self.kernel_size,
            )
            y = torch.einsum("gock,bgckt->bgot", weight_grouped, sampled_grouped)
            y = y.reshape(bsz, self.out_channels, seq_len)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        offsets = self._build_offsets(x)
        sampled = self._deform_sample(x, offsets)
        y = self._grouped_einsum_conv(sampled)

        offset_l1 = offsets.abs().mean()
        if offsets.shape[-1] > 1:
            offset_tv = (offsets[..., 1:] - offsets[..., :-1]).abs().mean()
        else:
            offset_tv = offsets.new_zeros(())
        if self.max_offset_scale > 0.0:
            max_offset = float(self.dilation) * float(self.max_offset_scale)
            saturation_ratio = (offsets.abs() >= (0.95 * max_offset)).to(dtype=offsets.dtype).mean()
        else:
            saturation_ratio = offsets.new_zeros(())
        self._last_offset_l1 = offset_l1
        self._last_offset_tv = offset_tv
        self._last_offset_saturation_ratio = saturation_ratio
        if not return_aux:
            return y

        aux: Dict[str, torch.Tensor] = {
            "offset_mean": offsets.mean().detach(),
            "offset_std": offsets.std(unbiased=False).detach(),
            "offset_max_abs": offsets.abs().amax().detach(),
            "offset_l1": offset_l1.detach(),
            "offset_tv": offset_tv.detach(),
            "offset_saturation_ratio": saturation_ratio.detach(),
        }
        return y, aux


class CausalMaxPool1d(nn.Module):
    """
    因果最大池化（只使用历史窗口）。

    维度变化：
    输入: [B, C, T]
    输出: [B, C, T]
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size 必须 > 0，当前={kernel_size}")
        self.kernel_size = int(kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播。
        输入: [B, C, T]
        输出: [B, C, T]
        """
        # 因果填充：只在左侧填充
        x_padded = F.pad(x, (self.kernel_size - 1, 0))
        return self.pool(x_padded)


# ===========================================================================
#  多尺度因果卷积（融合AMTC思想）
# ===========================================================================

class AdaptiveMultiScaleCausalConv1d(nn.Module):
    """
    融合AMTC思想的多尺度因果卷积。

    设计动机：
    1. 将多尺度分支能力直接内嵌到TCN块内，减少外置冗余结构。
    2. 保持因果性，适配TCN时间建模。
    3. 通过FFT估计主导周期做分支加权，增强尺度自适应。

    维度变化：
    输入: [B, C_in, T]
    输出: [B, C_out, T]

    内部结构：
    - N 个不同卷积核的因果分支（depthwise → pointwise），各输出 [B, C_out, T]
    - 可选池化分支
    - FFT估计主导周期，softmax加权各分支
    - 拼接后 1x1 投影融合
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Iterable[int],
        dilation: int,
        use_pool_branch: bool = True,
        fft_temperature: float = 2.0,
        use_fft_adaptive: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels 必须 > 0")
        if out_channels <= 0:
            raise ValueError("out_channels 必须 > 0")
        if dilation <= 0:
            raise ValueError("dilation 必须 > 0")
        if fft_temperature <= 0.0:
            raise ValueError("fft_temperature 必须 > 0")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_sizes = _to_kernel_tuple(kernel_sizes, "kernel_sizes")
        self.dilation = int(dilation)
        self.use_pool_branch = bool(use_pool_branch)
        self.fft_temperature = float(fft_temperature)
        self.use_fft_adaptive = bool(use_fft_adaptive)

        # 每个卷积核尺度一个因果分支：depthwise → BN → ReLU → pointwise → BN → ReLU
        self.conv_branches = nn.ModuleList(
            [
                nn.Sequential(
                    CausalConv1d(
                        in_channels=self.in_channels,
                        out_channels=self.in_channels,
                        kernel_size=kernel,
                        dilation=self.dilation,
                        groups=self.in_channels,
                    ),
                    nn.BatchNorm1d(self.in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.out_channels),
                    nn.ReLU(inplace=True),
                )
                for kernel in self.kernel_sizes
            ]
        )

        # 可选池化分支：用最大核做因果池化
        self.pool_branch = None
        pool_kernel = max(self.kernel_sizes)
        if self.use_pool_branch:
            self.pool_branch = nn.Sequential(
                CausalMaxPool1d(kernel_size=pool_kernel),
                nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.out_channels),
                nn.ReLU(inplace=True),
            )

        # 注册各分支的感受野大小，用于FFT自适应加权
        branch_fields = [int(kernel) * self.dilation for kernel in self.kernel_sizes]
        if self.use_pool_branch:
            branch_fields.append(pool_kernel * self.dilation)
        self.register_buffer(
            "branch_receptive_fields",
            torch.tensor(branch_fields, dtype=torch.float32),
            persistent=False,
        )

        # 可学习的分支偏置 logits
        self.branch_logits = nn.Parameter(torch.zeros(len(branch_fields)))
        # 拼接后 1x1 投影融合
        self.fuse_project = nn.Sequential(
            nn.Conv1d(
                in_channels=self.out_channels * len(branch_fields),
                out_channels=self.out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.out_channels),
        )

    @staticmethod
    def _estimate_dominant_period(x: torch.Tensor) -> torch.Tensor:
        """
        用FFT估计每个样本的主导周期。

        输入: x [B, C, T]
        输出: dominant_period [B]

        计算步骤：
        1. 去均值 → rFFT
        2. 忽略直流分量
        3. 取幅值最大频率 → 换算为周期
        """
        _, _, seq_len = x.shape
        if seq_len <= 1:
            return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

        # 使用float32做FFT，避免AMP/half在某些长度上的数值与内核限制问题。
        x_fft = x.float()
        x_centered = x_fft - x_fft.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(x_centered, dim=-1)
        amplitude = torch.abs(spectrum)
        if amplitude.shape[-1] > 0:
            amplitude[..., 0] = 0.0  # 忽略直流分量

        peak_index = torch.argmax(amplitude, dim=-1).clamp(min=1)
        dominant_period = float(seq_len) / peak_index.to(dtype=x_fft.dtype)
        dominant_period = dominant_period.clamp(min=1.0, max=float(seq_len))
        return dominant_period.mean(dim=1).to(dtype=x.dtype)

    def _build_branch_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        构建分支权重。
        输入: x [B, C, T]
        输出: weights [B, N_branch]，softmax归一化

        当 use_fft_adaptive=True 时，根据主导周期与各分支感受野的距离加权；
        否则只用可学习 logits。
        """
        batch_size = x.shape[0]
        base_logits = self.branch_logits.view(1, -1)
        if not self.use_fft_adaptive:
            return torch.softmax(base_logits, dim=-1).expand(batch_size, -1)

        dominant_period = self._estimate_dominant_period(x)  # [B]
        receptive_fields = self.branch_receptive_fields.to(device=x.device, dtype=x.dtype).view(1, -1)
        distance = torch.abs(dominant_period.unsqueeze(-1) - receptive_fields)
        logits = -distance / self.fft_temperature + base_logits
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播。
        输入: [B, C_in, T]
        输出: [B, C_out, T]
        """
        if x.ndim != 3:
            raise ValueError(f"输入必须是 [B, C, T]，实际={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"输入通道数不匹配，期望={self.in_channels}，实际={x.shape[1]}"
            )

        # 各分支加权求和
        branch_weights = self._build_branch_weights(x)  # [B, N_branch]
        weighted_outputs: List[torch.Tensor] = []
        branch_idx = 0

        for branch in self.conv_branches:
            branch_out = branch(x)  # [B, C_out, T]
            weight = branch_weights[:, branch_idx].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            weighted_outputs.append(branch_out * weight)
            branch_idx += 1

        if self.pool_branch is not None:
            pool_out = self.pool_branch(x)  # [B, C_out, T]
            weight = branch_weights[:, branch_idx].unsqueeze(-1).unsqueeze(-1)
            weighted_outputs.append(pool_out * weight)

        # 拼接所有分支 → 1x1 融合
        concat = torch.cat(weighted_outputs, dim=1)  # [B, C_out*N_branch, T]
        fused = self.fuse_project(concat)  # [B, C_out, T]
        return fused


# ===========================================================================
#  通道注意力 (Squeeze-Excitation)
# ===========================================================================

class ChannelSE1D(nn.Module):
    """
    1D 通道注意力（Squeeze-Excitation）。

    维度路径：
    - 输入 `x`: `[B, C, T]`
    - 时序汇聚 `z`: `[B, C]`
    - 通道权重 `w`: `[B, C]`
    - 输出 `y`: `[B, C, T]`
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels 必须 > 0")
        if reduction <= 0:
            raise ValueError("reduction 必须 > 0")
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        输入: [B, C, T]
        输出: (加权后的 [B, C, T], 通道权重 [B, C])
        """
        if x.ndim != 3:
            raise ValueError(f"SE 输入必须是 [B,C,T]，当前 {tuple(x.shape)}")
        z = x.mean(dim=-1)  # [B, C] — 全局时序汇聚
        w = torch.sigmoid(self.fc2(torch.relu(self.fc1(z))))  # [B, C]
        y = x * w.unsqueeze(-1)  # [B, C, T]
        return y, w


# ===========================================================================
#  TCN 残差块
# ===========================================================================

class TemporalBlock(nn.Module):
    """
    TCN残差块（双层因果卷积）。

    支持四种第一层卷积形式：
    1. 标准因果卷积（基础TCN）
    2. 融合AMTC能力的多尺度因果卷积
    3. 可变形因果卷积（Deformable 1D Causal Conv）
    4. Hybrid Deform + Multi-Scale 并行门控融合

    维度变化：
    输入: [B, C_in, T]
    输出: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_channel_se: bool = False,
        channel_se_reduction: int = 8,
        use_multi_scale_conv: bool = False,
        multi_scale_kernel_sizes: Iterable[int] = (3, 5, 7),
        multi_scale_use_pool_branch: bool = True,
        multi_scale_fft_temperature: float = 2.0,
        multi_scale_use_fft_adaptive: bool = True,
        use_hybrid_deform_multiscale: bool = False,
        hybrid_kernel_sizes: Iterable[int] = (3, 5, 7),
        hybrid_use_pool_branch: bool = True,
        hybrid_fft_temperature: float = 2.0,
        hybrid_use_fft_adaptive: bool = True,
        use_deformable_conv: bool = False,
        deformable_conv_mode: str = "conv1_only",
        deformable_causal_mode: str = "strict",
        deform_offset_kernel_size: int = 3,
        deform_group_mode: str = "per_channel",
        deform_channel_groups: Sequence[Sequence[int]] | None = None,
        deform_max_offset_scale: float = 0.0,
        deform_zero_init: bool = False,
        use_deform_conv_gate: bool = False,
        deform_conv_gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.use_multi_scale_conv = bool(use_multi_scale_conv)
        self.use_hybrid_deform_multiscale = bool(use_hybrid_deform_multiscale)
        self.use_deformable_conv = bool(use_deformable_conv)
        self.deformable_conv_mode = str(deformable_conv_mode).strip().lower()
        self.deformable_causal_mode = str(deformable_causal_mode).strip().lower()
        self.deform_offset_kernel_size = int(deform_offset_kernel_size)
        self.deform_group_mode = str(deform_group_mode).strip().lower()
        self.deform_channel_groups = (
            tuple(tuple(int(idx) for idx in group) for group in deform_channel_groups)
            if deform_channel_groups is not None
            else None
        )
        self.deform_max_offset_scale = float(deform_max_offset_scale)
        self.deform_zero_init = bool(deform_zero_init)
        self.use_deform_conv_gate = bool(use_deform_conv_gate)
        self.deform_conv_gate_bias_init = float(deform_conv_gate_bias_init)
        if self.use_hybrid_deform_multiscale and self.use_multi_scale_conv:
            raise ValueError(
                "use_hybrid_deform_multiscale 与 use_multi_scale_conv 不能同时开启"
            )
        if (not self.use_hybrid_deform_multiscale) and self.use_multi_scale_conv and self.use_deformable_conv:
            raise ValueError("use_multi_scale_conv 与 use_deformable_conv 不能同时开启")
        if self.use_hybrid_deform_multiscale and not self.use_deformable_conv:
            raise ValueError("use_hybrid_deform_multiscale 开启时必须同时开启 use_deformable_conv")
        if self.deformable_conv_mode not in {"conv1_only", "both"}:
            raise ValueError("deformable_conv_mode 必须是 conv1_only 或 both")
        if self.deformable_causal_mode not in {"strict", "relaxed"}:
            raise ValueError("deformable_causal_mode 必须是 strict 或 relaxed")
        if self.deform_offset_kernel_size <= 1 or self.deform_offset_kernel_size % 2 == 0:
            raise ValueError("deform_offset_kernel_size 必须是大于1的奇数")
        if self.use_deform_conv_gate and self.use_hybrid_deform_multiscale:
            raise ValueError("use_deform_conv_gate 当前不支持 hybrid deform 模式")

        # 第一层卷积：Hybrid / 多尺度 / 标准(含可变形)
        self.hybrid_deform_conv: DeformableCausalConv1d | None = None
        self.hybrid_multi_scale_conv: AdaptiveMultiScaleCausalConv1d | None = None
        self.hybrid_gate: nn.Sequential | None = None
        self.standard_conv1_for_gate: CausalConv1d | None = None
        self.deform_conv_gate_logit: nn.Parameter | None = None
        if self.use_hybrid_deform_multiscale:
            hybrid_kernel_sizes = _to_kernel_tuple(hybrid_kernel_sizes, "hybrid_kernel_sizes")
            if hybrid_fft_temperature <= 0.0:
                raise ValueError("hybrid_fft_temperature 必须 > 0")
            self.hybrid_deform_conv = DeformableCausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                causal_mode=self.deformable_causal_mode,
                offset_kernel_size=self.deform_offset_kernel_size,
                group_mode=self.deform_group_mode,
                channel_groups=self.deform_channel_groups,
                max_offset_scale=self.deform_max_offset_scale,
                zero_init=self.deform_zero_init,
            )
            self.hybrid_multi_scale_conv = AdaptiveMultiScaleCausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=hybrid_kernel_sizes,
                dilation=dilation,
                use_pool_branch=hybrid_use_pool_branch,
                fft_temperature=hybrid_fft_temperature,
                use_fft_adaptive=hybrid_use_fft_adaptive,
            )
            gate_hidden = max(out_channels // 8, 1)
            self.hybrid_gate = nn.Sequential(
                nn.Conv1d(out_channels * 2, gate_hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(gate_hidden, out_channels, kernel_size=1, bias=True),
            )
            self.multi_scale_conv = None
            self.conv1 = None
        elif self.use_multi_scale_conv:
            self.multi_scale_conv = AdaptiveMultiScaleCausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=multi_scale_kernel_sizes,
                dilation=dilation,
                use_pool_branch=multi_scale_use_pool_branch,
                fft_temperature=multi_scale_fft_temperature,
                use_fft_adaptive=multi_scale_use_fft_adaptive,
            )
            self.conv1 = None
        else:
            self.multi_scale_conv = None
            if self.use_deformable_conv:
                self.conv1 = DeformableCausalConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    causal_mode=self.deformable_causal_mode,
                    offset_kernel_size=self.deform_offset_kernel_size,
                    group_mode=self.deform_group_mode,
                    channel_groups=self.deform_channel_groups,
                    max_offset_scale=self.deform_max_offset_scale,
                    zero_init=self.deform_zero_init,
                )
                if self.use_deform_conv_gate:
                    self.standard_conv1_for_gate = CausalConv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    )
                    self.deform_conv_gate_logit = nn.Parameter(
                        torch.tensor(float(self.deform_conv_gate_bias_init))
                    )
            else:
                self.conv1 = CausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # 第二层：标准因果卷积 或 可变形因果卷积
        if self.use_deformable_conv and self.deformable_conv_mode == "both":
            self.conv2 = DeformableCausalConv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                causal_mode=self.deformable_causal_mode,
                offset_kernel_size=self.deform_offset_kernel_size,
                group_mode="per_channel",
                channel_groups=None,
                max_offset_scale=self.deform_max_offset_scale,
                zero_init=self.deform_zero_init,
            )
        else:
            self.conv2 = CausalConv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 残差连接：通道数不同时用 1x1 投影
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.layer_norm = nn.LayerNorm(out_channels)

        # 可选 ChannelSE 通道注意力
        self.channel_se = (
            ChannelSE1D(channels=out_channels, reduction=int(channel_se_reduction))
            if bool(use_channel_se)
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播。
        输入: [B, C_in, T]
        输出: [B, C_out, T]
        """
        # 第一层卷积
        deform_aux_stats: List[Dict[str, torch.Tensor]] = []
        hybrid_gate_mean: torch.Tensor | None = None
        if self.use_hybrid_deform_multiscale:
            if self.hybrid_deform_conv is None or self.hybrid_multi_scale_conv is None or self.hybrid_gate is None:
                raise RuntimeError("hybrid 第一层卷积未初始化")
            if return_aux:
                deform_out = self.hybrid_deform_conv(x, return_aux=True)
                y_deform = deform_out[0]
                deform_aux_stats.append(deform_out[1])
            else:
                y_deform = self.hybrid_deform_conv(x)
            y_multi_scale = self.hybrid_multi_scale_conv(x)
            gate = torch.sigmoid(self.hybrid_gate(torch.cat([y_deform, y_multi_scale], dim=1)))
            y = gate * y_deform + (1.0 - gate) * y_multi_scale
            hybrid_gate_mean = gate.mean().detach()
        elif self.use_multi_scale_conv and self.multi_scale_conv is not None:
            y = self.multi_scale_conv(x)
        else:
            if self.conv1 is None:
                raise RuntimeError("conv1 未初始化")
            if isinstance(self.conv1, DeformableCausalConv1d):
                if self.standard_conv1_for_gate is not None and self.deform_conv_gate_logit is not None:
                    if return_aux:
                        conv1_out = self.conv1(x, return_aux=True)
                        y_deform = conv1_out[0]
                        deform_aux_stats.append(conv1_out[1])
                    else:
                        y_deform = self.conv1(x)
                    y_standard = self.standard_conv1_for_gate(x)
                    gate = torch.sigmoid(self.deform_conv_gate_logit).view(1, 1, 1)
                    y = gate * y_deform + (1.0 - gate) * y_standard
                    hybrid_gate_mean = gate.squeeze().detach()
                elif return_aux:
                    conv1_out = self.conv1(x, return_aux=True)
                    y = conv1_out[0]
                    deform_aux_stats.append(conv1_out[1])
                else:
                    y = self.conv1(x)
            else:
                y = self.conv1(x)
        y = self.relu1(y)
        y = self.dropout1(y)

        # 第二层卷积
        if isinstance(self.conv2, DeformableCausalConv1d):
            if return_aux:
                conv2_out = self.conv2(y, return_aux=True)
                y = conv2_out[0]
                deform_aux_stats.append(conv2_out[1])
            else:
                y = self.conv2(y)
        else:
            y = self.conv2(y)
        y = self.relu2(y)
        y = self.dropout2(y)

        # 残差连接 + LayerNorm
        res = x if self.downsample is None else self.downsample(x)
        out = y + res
        out = out.transpose(1, 2)   # [B, T, C] for LayerNorm
        out = self.layer_norm(out)
        out = out.transpose(1, 2)   # [B, C, T]

        # 可选 ChannelSE
        channel_attn_weights = None
        if self.channel_se is not None:
            out, channel_attn_weights = self.channel_se(out)

        if not return_aux:
            return out

        aux: Dict[str, torch.Tensor] = {}
        if channel_attn_weights is not None:
            aux["channel_attn_weights"] = channel_attn_weights
        if len(deform_aux_stats) > 0:
            aux["deform_offset_mean"] = torch.stack(
                [stats["offset_mean"] for stats in deform_aux_stats], dim=0
            ).mean(dim=0)
            aux["deform_offset_std"] = torch.stack(
                [stats["offset_std"] for stats in deform_aux_stats], dim=0
            ).mean(dim=0)
            aux["deform_offset_max_abs"] = torch.stack(
                [stats["offset_max_abs"] for stats in deform_aux_stats], dim=0
            ).amax(dim=0)
            aux["deform_offset_l1"] = torch.stack(
                [stats["offset_l1"] for stats in deform_aux_stats], dim=0
            ).sum(dim=0)
            aux["deform_offset_tv"] = torch.stack(
                [stats["offset_tv"] for stats in deform_aux_stats], dim=0
            ).sum(dim=0)
            aux["deform_offset_saturation_ratio"] = torch.stack(
                [stats["offset_saturation_ratio"] for stats in deform_aux_stats], dim=0
            ).mean(dim=0)
        if hybrid_gate_mean is not None:
            aux["hybrid_gate_mean"] = hybrid_gate_mean
        return out, aux

    def get_deform_offset_l1(self) -> torch.Tensor:
        total = self.layer_norm.weight.new_zeros(())
        if self.hybrid_deform_conv is not None:
            total = total + self.hybrid_deform_conv.get_last_offset_l1()
        if isinstance(self.conv1, DeformableCausalConv1d):
            total = total + self.conv1.get_last_offset_l1()
        if isinstance(self.conv2, DeformableCausalConv1d):
            total = total + self.conv2.get_last_offset_l1()
        return total

    def get_deform_offset_tv(self) -> torch.Tensor:
        total = self.layer_norm.weight.new_zeros(())
        if self.hybrid_deform_conv is not None:
            total = total + self.hybrid_deform_conv.get_last_offset_tv()
        if isinstance(self.conv1, DeformableCausalConv1d):
            total = total + self.conv1.get_last_offset_tv()
        if isinstance(self.conv2, DeformableCausalConv1d):
            total = total + self.conv2.get_last_offset_tv()
        return total

    def get_deform_offset_saturation_ratio(self) -> torch.Tensor:
        stats: list[torch.Tensor] = []
        if self.hybrid_deform_conv is not None:
            stats.append(self.hybrid_deform_conv.get_last_offset_saturation_ratio())
        if isinstance(self.conv1, DeformableCausalConv1d):
            stats.append(self.conv1.get_last_offset_saturation_ratio())
        if isinstance(self.conv2, DeformableCausalConv1d):
            stats.append(self.conv2.get_last_offset_saturation_ratio())
        if len(stats) == 0:
            return self.layer_norm.weight.new_zeros(())
        return torch.stack(stats, dim=0).mean()

    def iter_deform_offset_parameters(self):
        if self.hybrid_deform_conv is not None:
            yield from self.hybrid_deform_conv.iter_offset_parameters()
        if isinstance(self.conv1, DeformableCausalConv1d):
            yield from self.conv1.iter_offset_parameters()
        if isinstance(self.conv2, DeformableCausalConv1d):
            yield from self.conv2.iter_offset_parameters()


# ===========================================================================
#  TCN 主体网络
# ===========================================================================

class TemporalConvNet(nn.Module):
    """
    TCN主体网络（精简版）。

    结构：
    input -> (optional LWPT) ->
    [TemporalBlock x N + optional layerwise TF fusion] ->
    masked pooling -> optional global branch weighting -> classifier

    精简后保留的模块：
    - LWPT 前端（可学习频带分解）
    - Fused-AMTC-TCN（多尺度因果卷积内嵌 TCN 块）
    - ChannelSE（通道注意力）
    - TF2D 时频分支 + LayerwiseFusion + GlobalBranchWeighting
    - MaskedMeanPooling
    """

    def __init__(
        self,
        # ── 核心参数 ──
        input_dim: int = None,
        num_classes: int = None,
        tcn_channels: int = None,
        tcn_num_levels: int = None,
        tcn_kernel_size: int = None,
        tcn_dropout: float = None,
        # ── LWPT 前端 ──
        use_lwpt_frontend: bool | None = None,
        lwpt_num_bands: int | None = None,
        lwpt_kernel_sizes: tuple[int, ...] | list[int] | None = None,
        lwpt_dropout: float | None = None,
        # ── ChannelSE ──
        use_channel_se: bool | None = None,
        channel_se_reduction: int | None = None,
        # ── Fused-AMTC-TCN ──
        use_fused_amtc_tcn: bool | None = None,
        fused_tcn_kernel_sizes: tuple[int, ...] | list[int] | None = None,
        fused_tcn_use_pool_branch: bool | None = None,
        fused_tcn_fft_temperature: float | None = None,
        fused_tcn_use_fft_adaptive: bool | None = None,
        # ── TF2D 时频分支 ──
        use_tf2d_branch: bool | None = None,
        tf2d_input_from_raw: bool | None = None,
        tf2d_n_fft: tuple[int, ...] | list[int] | None = None,
        tf2d_hop_lengths: tuple[int, ...] | list[int] | None = None,
        tf2d_win_lengths: tuple[int, ...] | list[int] | None = None,
        tf2d_log_scale: bool | None = None,
        tf2d_branch_channels: int | None = None,
        tf2d_pool_freq_bins: int | None = None,
        tf2d_pool_time_bins: int | None = None,
        # ── 融合 ──
        use_layerwise_tf_fusion: bool | None = None,
        tf_fusion_alpha_init: float | None = None,
        tf_fusion_dropout: float | None = None,
        use_global_branch_weighting: bool | None = None,
        # ── 健康掩码 ──
        concat_health_mask_channels: bool | None = None,
    ) -> None:
        """
        初始化TCN网络。
        所有参数若传入None，则从CFG读取默认值。
        """
        super().__init__()

        # ── 读取核心参数（None 时从 CFG 回退） ──
        concat_mask = (
            bool(concat_health_mask_channels)
            if concat_health_mask_channels is not None
            else bool(getattr(CFG, "concat_health_mask_channels", False))
        )
        base_raw_input_dim = int(getattr(CFG, "input_dim", 19))
        cross_sensor_residual_channels = (
            int(getattr(CFG, "cross_sensor_residual_channels", 9))
            if bool(getattr(CFG, "use_cross_sensor_residuals", False))
            else 0
        )
        raw_sensor_input_dim = base_raw_input_dim + cross_sensor_residual_channels
        mask_input_dim = base_raw_input_dim if concat_mask else 0
        default_input_dim = raw_sensor_input_dim + mask_input_dim
        self.input_dim = int(input_dim) if input_dim is not None else default_input_dim
        self.num_classes = num_classes if num_classes is not None else CFG.num_classes
        self.tcn_channels = tcn_channels if tcn_channels is not None else CFG.tcn_channels
        self.tcn_num_levels = tcn_num_levels if tcn_num_levels is not None else CFG.tcn_num_levels
        tcn_kernel_size = tcn_kernel_size if tcn_kernel_size is not None else CFG.tcn_kernel_size
        tcn_dropout = tcn_dropout if tcn_dropout is not None else CFG.tcn_dropout

        # ── LWPT 前端参数 ──
        self.use_lwpt_frontend = (
            bool(use_lwpt_frontend)
            if use_lwpt_frontend is not None
            else bool(getattr(CFG, "use_lwpt_frontend", False))
        )
        self.lwpt_num_bands = (
            int(lwpt_num_bands)
            if lwpt_num_bands is not None
            else int(getattr(CFG, "lwpt_num_bands", 4))
        )
        self.lwpt_kernel_sizes = (
            tuple(int(k) for k in lwpt_kernel_sizes)
            if lwpt_kernel_sizes is not None
            else tuple(int(k) for k in getattr(CFG, "lwpt_kernel_sizes", (3, 5, 7, 9)))
        )
        self.lwpt_dropout = (
            float(lwpt_dropout)
            if lwpt_dropout is not None
            else float(getattr(CFG, "lwpt_dropout", 0.1))
        )

        # ── ChannelSE 参数 ──
        self.use_channel_se = (
            bool(use_channel_se)
            if use_channel_se is not None
            else bool(getattr(CFG, "use_channel_se", False))
        )
        self.channel_se_reduction = (
            int(channel_se_reduction)
            if channel_se_reduction is not None
            else int(getattr(CFG, "channel_se_reduction", 8))
        )

        # ── Fused-AMTC-TCN 参数 ──
        self.use_fused_amtc_tcn = (
            bool(use_fused_amtc_tcn)
            if use_fused_amtc_tcn is not None
            else bool(getattr(CFG, "use_fused_amtc_tcn", False))
        )
        self.fused_tcn_kernel_sizes = (
            tuple(int(k) for k in fused_tcn_kernel_sizes)
            if fused_tcn_kernel_sizes is not None
            else tuple(int(k) for k in getattr(CFG, "fused_tcn_kernel_sizes", (3, 5, 7)))
        )
        self.fused_tcn_use_pool_branch = (
            bool(fused_tcn_use_pool_branch)
            if fused_tcn_use_pool_branch is not None
            else bool(getattr(CFG, "fused_tcn_use_pool_branch", True))
        )
        self.fused_tcn_fft_temperature = (
            float(fused_tcn_fft_temperature)
            if fused_tcn_fft_temperature is not None
            else float(getattr(CFG, "fused_tcn_fft_temperature", 2.0))
        )
        self.fused_tcn_use_fft_adaptive = (
            bool(fused_tcn_use_fft_adaptive)
            if fused_tcn_use_fft_adaptive is not None
            else bool(getattr(CFG, "fused_tcn_use_fft_adaptive", True))
        )

        # ── TF2D 时频分支参数 ──
        self.use_tf2d_branch = (
            bool(use_tf2d_branch)
            if use_tf2d_branch is not None
            else bool(getattr(CFG, "use_tf2d_branch", False))
        )
        self.tf2d_input_from_raw = (
            bool(tf2d_input_from_raw)
            if tf2d_input_from_raw is not None
            else bool(getattr(CFG, "tf2d_input_from_raw", True))
        )
        self.tf2d_n_fft = (
            tuple(int(v) for v in tf2d_n_fft)
            if tf2d_n_fft is not None
            else tuple(int(v) for v in getattr(CFG, "tf2d_n_fft", (16, 32, 48)))
        )
        self.tf2d_hop_lengths = (
            tuple(int(v) for v in tf2d_hop_lengths)
            if tf2d_hop_lengths is not None
            else tuple(int(v) for v in getattr(CFG, "tf2d_hop_lengths", (4, 8, 12)))
        )
        self.tf2d_win_lengths = (
            tuple(int(v) for v in tf2d_win_lengths)
            if tf2d_win_lengths is not None
            else tuple(int(v) for v in getattr(CFG, "tf2d_win_lengths", (16, 32, 48)))
        )
        self.tf2d_log_scale = (
            bool(tf2d_log_scale)
            if tf2d_log_scale is not None
            else bool(getattr(CFG, "tf2d_log_scale", True))
        )
        self.tf2d_branch_channels = (
            int(tf2d_branch_channels)
            if tf2d_branch_channels is not None
            else int(getattr(CFG, "tf2d_branch_channels", 64))
        )
        self.tf2d_pool_freq_bins = (
            int(tf2d_pool_freq_bins)
            if tf2d_pool_freq_bins is not None
            else int(getattr(CFG, "tf2d_pool_freq_bins", 8))
        )
        self.tf2d_pool_time_bins = (
            int(tf2d_pool_time_bins)
            if tf2d_pool_time_bins is not None
            else int(getattr(CFG, "tf2d_pool_time_bins", 24))
        )

        # ── 融合参数 ──
        self.use_layerwise_tf_fusion = (
            bool(use_layerwise_tf_fusion)
            if use_layerwise_tf_fusion is not None
            else bool(getattr(CFG, "use_layerwise_tf_fusion", True))
        )
        self.tf_fusion_alpha_init = (
            float(tf_fusion_alpha_init)
            if tf_fusion_alpha_init is not None
            else float(getattr(CFG, "tf_fusion_alpha_init", 0.0))
        )
        self.tf_fusion_dropout = (
            float(tf_fusion_dropout)
            if tf_fusion_dropout is not None
            else float(getattr(CFG, "tf_fusion_dropout", 0.1))
        )
        self.use_global_branch_weighting = (
            bool(use_global_branch_weighting)
            if use_global_branch_weighting is not None
            else bool(getattr(CFG, "use_global_branch_weighting", True))
        )

        # ── 参数校验 ──
        if self.use_lwpt_frontend:
            if self.lwpt_num_bands < 1:
                raise ValueError("lwpt_num_bands 必须 >= 1")
            self.lwpt_kernel_sizes = _to_kernel_tuple(self.lwpt_kernel_sizes, "lwpt_kernel_sizes")
            if not (0.0 <= self.lwpt_dropout < 1.0):
                raise ValueError("lwpt_dropout 必须在 [0,1) 范围")
        if self.use_channel_se and self.channel_se_reduction <= 0:
            raise ValueError("channel_se_reduction 必须 > 0")
        if self.use_fused_amtc_tcn:
            self.fused_tcn_kernel_sizes = _to_kernel_tuple(
                self.fused_tcn_kernel_sizes,
                "fused_tcn_kernel_sizes",
            )
            if self.fused_tcn_fft_temperature <= 0.0:
                raise ValueError("fused_tcn_fft_temperature 必须 > 0")
        if self.use_tf2d_branch:
            if not (len(self.tf2d_n_fft) == len(self.tf2d_hop_lengths) == len(self.tf2d_win_lengths)):
                raise ValueError("tf2d_n_fft/tf2d_hop_lengths/tf2d_win_lengths 长度必须一致")
            if self.tf2d_branch_channels <= 0:
                raise ValueError("tf2d_branch_channels 必须 > 0")
            if self.tf2d_pool_freq_bins <= 0 or self.tf2d_pool_time_bins <= 0:
                raise ValueError("tf2d_pool_freq_bins/tf2d_pool_time_bins 必须 > 0")
            if not (0.0 <= self.tf_fusion_dropout < 1.0):
                raise ValueError("tf_fusion_dropout 必须在 [0, 1) 范围内")

        # ══════════════════════════════════════════════════════════════
        #  构建子模块
        # ══════════════════════════════════════════════════════════════

        # 1) LWPT 前端
        if self.use_lwpt_frontend:
            self.lwpt_frontend = LearnableWaveletPacketFrontend(
                in_channels=self.input_dim,
                num_bands=self.lwpt_num_bands,
                kernel_sizes=self.lwpt_kernel_sizes,
                dropout=self.lwpt_dropout,
            )
        else:
            self.lwpt_frontend = nn.Identity()

        tcn_input_channels = self.input_dim

        # 2) TCN 骨干（含可选 Fused-AMTC 多尺度卷积 + ChannelSE）
        layers: List[TemporalBlock] = []
        in_ch = tcn_input_channels
        for level in range(self.tcn_num_levels):
            dilation = 2**level
            out_ch = self.tcn_channels
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    dropout=tcn_dropout,
                    use_channel_se=self.use_channel_se,
                    channel_se_reduction=self.channel_se_reduction,
                    use_multi_scale_conv=self.use_fused_amtc_tcn,
                    multi_scale_kernel_sizes=self.fused_tcn_kernel_sizes,
                    multi_scale_use_pool_branch=self.fused_tcn_use_pool_branch,
                    multi_scale_fft_temperature=self.fused_tcn_fft_temperature,
                    multi_scale_use_fft_adaptive=self.fused_tcn_use_fft_adaptive,
                )
            )
            in_ch = out_ch
        self.network = nn.ModuleList(layers)

        # 3) TF2D 时频分支（可选）
        if self.use_tf2d_branch:
            tf_in_channels = self.input_dim if self.tf2d_input_from_raw else tcn_input_channels
            self.timefreq_stft = MultiResolutionSTFT(
                n_fft=self.tf2d_n_fft,
                hop_lengths=self.tf2d_hop_lengths,
                win_lengths=self.tf2d_win_lengths,
                log_scale=self.tf2d_log_scale,
            )
            self.timefreq_encoder = TimeFreqEncoder2D(
                in_channels=tf_in_channels,
                num_scales=len(self.tf2d_n_fft),
                out_channels=self.tcn_channels,
                scale_branch_channels=self.tf2d_branch_channels,
                pool_freq_bins=self.tf2d_pool_freq_bins,
                pool_time_bins=self.tf2d_pool_time_bins,
            )
            self.timefreq_align = TimeFreqAlign1D()
            self.timefreq_spec_augment = TimeFreqSpecAugmentor(
                enable=bool(getattr(CFG, "augment_freq_domain_enable", False)),
                time_mask_enable=bool(getattr(CFG, "augment_freq_time_mask_enable", True)),
                time_mask_prob=float(getattr(CFG, "augment_freq_time_mask_prob", 0.35)),
                time_mask_width_range=tuple(
                    int(v) for v in getattr(CFG, "augment_freq_time_mask_width_range", (2, 6))
                ),
                freq_mask_enable=bool(getattr(CFG, "augment_freq_freq_mask_enable", True)),
                freq_mask_prob=float(getattr(CFG, "augment_freq_freq_mask_prob", 0.35)),
                freq_mask_width_range=tuple(
                    int(v) for v in getattr(CFG, "augment_freq_freq_mask_width_range", (1, 2))
                ),
                spectral_tilt_enable=bool(getattr(CFG, "augment_freq_spectral_tilt_enable", True)),
                spectral_tilt_prob=float(getattr(CFG, "augment_freq_spectral_tilt_prob", 0.2)),
                spectral_tilt_db_range=float(
                    getattr(CFG, "augment_freq_spectral_tilt_db_range", 1.5)
                ),
            )
        else:
            self.timefreq_stft = None
            self.timefreq_encoder = None
            self.timefreq_align = None
            self.timefreq_spec_augment = None

        # 4) 层级门控融合（TF2D 分支与 TCN 主干的逐层融合）
        if self.use_tf2d_branch and self.use_layerwise_tf_fusion:
            self.fusion_gates = nn.ModuleList(
                [
                    LayerwiseFusionGate(
                        channels=self.tcn_channels,
                        alpha_init=self.tf_fusion_alpha_init,
                        dropout=self.tf_fusion_dropout,
                    )
                    for _ in range(self.tcn_num_levels)
                ]
            )
        else:
            self.fusion_gates = nn.ModuleList([])

        # 5) 全局分支加权（时域+频域特征的自适应加权）
        if self.use_tf2d_branch and self.use_global_branch_weighting:
            self.branch_weight_mlp = nn.Sequential(
                nn.Linear(self.tcn_channels * 2, self.tcn_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.tcn_channels, 2),
            )
        else:
            self.branch_weight_mlp = None

        # 6) 分类器
        self.classifier = nn.Linear(self.tcn_channels, self.num_classes)

    # ==================================================================
    #  内部辅助方法
    # ==================================================================

    def _build_timefreq_sequence(
        self,
        raw_input: torch.Tensor,
        frontend_input: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor | None:
        """
        构建时频分支并对齐到TCN时间轴。

        输入:
        raw_input:      [B, C_raw, T]
        frontend_input: [B, C_front, T]
        输出:
        tf_seq: [B, C_tcn, T] 或 None
        """
        if not self.use_tf2d_branch:
            return None
        if self.timefreq_stft is None or self.timefreq_encoder is None or self.timefreq_align is None:
            raise RuntimeError("时频分支模块未初始化")
        tf_source = raw_input if self.tf2d_input_from_raw else frontend_input
        specs = self.timefreq_stft(tf_source)                                 # List[[B, C, F_i, T_i]]
        if self.timefreq_spec_augment is not None:
            specs = self.timefreq_spec_augment(specs)
        tf_seq_short = self.timefreq_encoder(specs)                           # [B, C_tcn, T_pool]
        tf_seq = self.timefreq_align(tf_seq_short, target_length=target_length)  # [B, C_tcn, T]
        return tf_seq

    def _forward_tcn_with_optional_layerwise_fusion(
        self,
        x_in: torch.Tensor,
        tf_seq: torch.Tensor | None,
        collect_aux: bool,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        """
        逐层执行TCN并可选注入分层门控融合。

        输入:
        x_in: [B, C_in, T]
        tf_seq: [B, C_tcn, T] 或 None
        输出:
        y: [B, C_tcn, T]
        """
        y = x_in
        layer_gate_means: List[float] = []
        layer_gate_mins: List[float] = []
        layer_gate_maxs: List[float] = []
        layer_alpha_values: List[float] = []
        channel_attn_per_layer: List[torch.Tensor] = []

        for idx, block in enumerate(self.network):
            if collect_aux:
                block_out = block(y, return_aux=True)
                y = block_out[0]
                block_aux = block_out[1]
                if "channel_attn_weights" in block_aux:
                    channel_attn_per_layer.append(block_aux["channel_attn_weights"])
            else:
                y = block(y)

            # 可选层级门控融合
            if tf_seq is not None and self.use_layerwise_tf_fusion and idx < len(self.fusion_gates):
                y, gate_stats = self.fusion_gates[idx](y, tf_seq)
                if collect_aux:
                    layer_gate_means.append(float(gate_stats["gate_mean"]))
                    layer_gate_mins.append(float(gate_stats["gate_min"]))
                    layer_gate_maxs.append(float(gate_stats["gate_max"]))
                    layer_alpha_values.append(float(gate_stats["alpha"]))

        aux: Dict[str, object] = {}
        if collect_aux:
            aux["layer_gate_means"] = layer_gate_means
            aux["layer_gate_mins"] = layer_gate_mins
            aux["layer_gate_maxs"] = layer_gate_maxs
            aux["layer_alpha_values"] = layer_alpha_values
            if len(channel_attn_per_layer) > 0:
                stacked = torch.stack(channel_attn_per_layer, dim=0)
                aux["channel_attn_weights_summary"] = stacked.mean(dim=0).detach().cpu()
            else:
                aux["channel_attn_weights_summary"] = None
        return y, aux

    def _extract_sequence_features_with_aux(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        collect_aux: bool,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        """
        提取分类前序列特征，并按需返回融合过程辅助信息。

        数据流：
        x [B, T, C] → transpose → LWPT → TCN blocks (+ TF2D fusion) → pooling → pooled [B, C_tcn]
        """
        if x.ndim != 3:
            raise ValueError(f"输入必须是 [B, T, C]，实际={tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入通道数不匹配，期望={self.input_dim}，实际={x.shape[-1]}"
            )
        if lengths.ndim != 1:
            raise ValueError(f"lengths 必须是 [B]，实际={tuple(lengths.shape)}")
        if lengths.shape[0] != x.shape[0]:
            raise ValueError(
                f"lengths 与 batch 不匹配，lengths={lengths.shape[0]} batch={x.shape[0]}"
            )

        raw_input = x.transpose(1, 2)              # [B, C_raw, T]
        x_in = self.lwpt_frontend(raw_input)        # [B, C_raw, T]（LWPT 保形）

        # 构建时频分支（可选）
        tf_seq = self._build_timefreq_sequence(
            raw_input=raw_input,
            frontend_input=x_in,
            target_length=x_in.shape[-1],
        )

        # TCN 主干 + 可选层级融合
        y, layer_aux = self._forward_tcn_with_optional_layerwise_fusion(
            x_in=x_in,
            tf_seq=tf_seq,
            collect_aux=collect_aux,
        )

        # 时序池化：Masked Mean Pooling
        y_seq = y.transpose(1, 2)                   # [B, T, C_tcn]
        pooled_time = masked_mean_pooling(y_seq, lengths=lengths)  # [B, C_tcn]
        pooled = pooled_time
        branch_weights = None

        # 全局分支加权（时域 + 频域特征的自适应加权）
        if tf_seq is not None:
            tf_seq_tokens = tf_seq.transpose(1, 2)  # [B, T, C_tcn]
            pooled_freq = masked_mean_pooling(tf_seq_tokens, lengths=lengths)  # [B, C_tcn]
            if self.branch_weight_mlp is not None:
                weight_logits = self.branch_weight_mlp(torch.cat([pooled_time, pooled_freq], dim=1))
                branch_weights = torch.softmax(weight_logits, dim=-1)  # [B, 2]
                pooled = (
                    branch_weights[:, 0:1] * pooled_time
                    + branch_weights[:, 1:2] * pooled_freq
                )
            else:
                pooled = 0.5 * (pooled_time + pooled_freq)
                branch_weights = torch.full(
                    (pooled_time.shape[0], 2),
                    fill_value=0.5,
                    dtype=pooled_time.dtype,
                    device=pooled_time.device,
                )

        aux: Dict[str, object] = {}
        if collect_aux:
            aux.update(layer_aux)
            aux["tf_seq_shape"] = tuple(tf_seq.shape) if tf_seq is not None else None
            aux["branch_weights"] = (
                branch_weights.detach().cpu() if branch_weights is not None else None
            )
            aux["pooled_shape"] = tuple(pooled.shape)
        return pooled, aux

    def _extract_sequence_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取时序特征（不经过分类头）。
        输入: x [B, T, C], lengths [B]
        输出: pooled [B, C_tcn]
        """
        pooled, _ = self._extract_sequence_features_with_aux(
            x=x,
            lengths=lengths,
            collect_aux=False,
        )
        return pooled

    # ==================================================================
    #  公开接口
    # ==================================================================

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。
        输入:
        x: [B, T, C]
        lengths: [B]
        输出:
        logits: [B, num_classes]
        """
        pooled = self._extract_sequence_features(x=x, lengths=lengths)
        logits = self.classifier(pooled)
        return logits

    def forward_with_aux(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        """
        带辅助统计信息的前向传播。
        用途：
        1. 分析层级门控融合行为。
        2. 分析决策级时域/时频分支权重。
        """
        pooled, aux = self._extract_sequence_features_with_aux(
            x=x,
            lengths=lengths,
            collect_aux=True,
        )
        logits = self.classifier(pooled)
        # 供辅助特征损失（如 SupCon/CenterLoss）复用，避免二次前向提特征。
        aux["pooled_features"] = pooled
        aux["logits_shape"] = tuple(logits.shape)
        return logits, aux

    def freeze_feature_extractor(self, freeze: bool = True) -> None:
        """
        冻结/解冻特征提取层（用于迁移学习微调）。
        覆盖范围：
        1. LWPT 前端。
        2. TCN 主干（含 Fused-AMTC-TCN 多尺度卷积分支）。
        3. 时频分支与层级融合门控、全局分支权重模块。
        """
        feature_modules: List[nn.Module] = [
            self.lwpt_frontend,
            self.network,
            self.fusion_gates,
        ]
        if self.timefreq_stft is not None:
            feature_modules.append(self.timefreq_stft)
        if self.timefreq_encoder is not None:
            feature_modules.append(self.timefreq_encoder)
        if self.timefreq_align is not None:
            feature_modules.append(self.timefreq_align)
        if self.timefreq_spec_augment is not None:
            feature_modules.append(self.timefreq_spec_augment)
        if self.branch_weight_mlp is not None:
            feature_modules.append(self.branch_weight_mlp)

        for module in feature_modules:
            for param in module.parameters():
                param.requires_grad = not freeze

    def get_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取分类前特征。
        输入: [B, T, C]
        输出: [B, C_tcn]
        """
        return self._extract_sequence_features(x=x, lengths=lengths)
