"""
simplified_fft_lwpt_se_tcn.py

简化版网络（架构改进版）：
1. 时域分支：LWPT 前端（可选）+ TCN（因果时序建模）。
2. 频域分支：原始输入 FFT 幅值 -> 非因果 Conv1d 投影分支。
3. 注意力：时域/频域分支各自独立通道注意力（Separated SE）。
4. 融合：分支各自池化后做 Late Fusion，再送分类头。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from essence_forge.core.runtime_config import CFG
from essence_forge.core.channel_layout import channel_names_from_specs
from essence_forge.core.models.lwpt import LearnableWaveletPacketFrontend
from essence_forge.core.models.masking import masked_max_pooling, masked_mean_pooling
from essence_forge.core.models.sensor_group_attention import SensorGroupAttention
from essence_forge.core.models.tcn import ChannelSE1D, TemporalBlock


def _to_kernel_tuple(kernel_sizes: Iterable[int], name: str) -> Tuple[int, ...]:
    kernels = tuple(int(k) for k in kernel_sizes)
    if len(kernels) == 0:
        raise ValueError(f"{name} 不能为空")
    for kernel in kernels:
        if kernel <= 1 or kernel % 2 == 0:
            raise ValueError(f"{name} 中卷积核必须是大于1的奇数，当前={kernel}")
    return kernels


class SimplifiedFftLwptSeTCN(nn.Module):
    """
    输入: x [B, T, C]
    流程:
    - raw_input = transpose(x) -> [B, C, T]
    - time_feat = LWPT(raw_input) -> [B, C, T] -> TCN -> pooled_time [B, C_tcn]
    - fft_mag = abs(FFT(raw_input)) -> [B, C, T] -> non-causal Conv1d -> pooled_freq [B, C_freq]
    - late_fused = cat([pooled_time, pooled_freq]) -> [B, C_tcn + C_freq]
    - late_fusion_proj -> [B, C_tcn]
    - classifier -> [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int | None = None,
        num_classes: int | None = None,
        tcn_channels: int | None = None,
        tcn_num_levels: int | None = None,
        tcn_kernel_size: int | None = None,
        tcn_dropout: float | None = None,
        use_lwpt_frontend: bool | None = None,
        lwpt_num_bands: int | None = None,
        lwpt_kernel_sizes: tuple[int, ...] | list[int] | None = None,
        lwpt_dropout: float | None = None,
        use_channel_se: bool | None = None,
        channel_se_reduction: int | None = None,
        use_mixed_pooling: bool | None = None,
        freq_branch_channels: int | None = None,
        freq_branch_kernel_size: int | None = None,
        freq_branch_dropout: float | None = None,
        use_freq_branch_se: bool | None = None,
        freq_branch_se_reduction: int | None = None,
        concat_health_mask_channels: bool | None = None,
        use_sensor_group_attention: bool | None = None,
        fft_mag_log1p: bool | None = None,
        fft_mag_norm: str | None = None,
        use_deformable_tcn: bool | None = None,
        deformable_conv_mode: str | None = None,
        deformable_causal_mode: str | None = None,
        deform_offset_kernel_size: int | None = None,
        deform_offset_l1_weight: float | None = None,
        use_hybrid_deform_multiscale_tcn: bool | None = None,
        hybrid_tcn_kernel_sizes: tuple[int, ...] | list[int] | None = None,
        use_levelwise_tcn_aggregation: bool | None = None,
        use_uncertainty_temporal_pooling: bool | None = None,
    ) -> None:
        super().__init__()

        concat_mask = (
            bool(concat_health_mask_channels)
            if concat_health_mask_channels is not None
            else bool(getattr(CFG, "concat_health_mask_channels", False))
        )
        self.base_raw_input_dim = int(getattr(CFG, "input_dim", 19))
        self.base_raw_channel_names = channel_names_from_specs(
            getattr(CFG, "channels", ())
        )
        self.cross_sensor_residual_channels = (
            int(getattr(CFG, "cross_sensor_residual_channels", 9))
            if bool(getattr(CFG, "use_cross_sensor_residuals", False))
            else 0
        )
        self.residual_channel_names = tuple(
            str(name) for name in getattr(CFG, "cross_sensor_residual_channel_names", ())
        )[: self.cross_sensor_residual_channels]
        self.raw_sensor_channel_names = (
            self.base_raw_channel_names + self.residual_channel_names
        )
        self.raw_sensor_input_dim = self.base_raw_input_dim + self.cross_sensor_residual_channels
        self.health_mask_input_dim = self.base_raw_input_dim if concat_mask else 0
        default_input_dim = self.raw_sensor_input_dim + self.health_mask_input_dim
        self.input_dim = int(input_dim) if input_dim is not None else default_input_dim
        self.num_classes = int(num_classes) if num_classes is not None else int(CFG.num_classes)
        self.tcn_channels = int(tcn_channels) if tcn_channels is not None else int(CFG.tcn_channels)
        self.tcn_num_levels = int(tcn_num_levels) if tcn_num_levels is not None else int(CFG.tcn_num_levels)
        self.tcn_kernel_size = int(tcn_kernel_size) if tcn_kernel_size is not None else int(CFG.tcn_kernel_size)
        self.tcn_dropout = float(tcn_dropout) if tcn_dropout is not None else float(CFG.tcn_dropout)

        self.use_lwpt_frontend = (
            bool(use_lwpt_frontend)
            if use_lwpt_frontend is not None
            else bool(getattr(CFG, "use_lwpt_frontend", True))
        )
        self.lwpt_num_bands = (
            int(lwpt_num_bands) if lwpt_num_bands is not None else int(getattr(CFG, "lwpt_num_bands", 4))
        )
        self.lwpt_kernel_sizes = (
            tuple(int(k) for k in lwpt_kernel_sizes)
            if lwpt_kernel_sizes is not None
            else tuple(int(k) for k in getattr(CFG, "lwpt_kernel_sizes", (3, 5, 7, 9)))
        )
        self.lwpt_dropout = (
            float(lwpt_dropout) if lwpt_dropout is not None else float(getattr(CFG, "lwpt_dropout", 0.1))
        )

        self.use_channel_se = (
            bool(use_channel_se)
            if use_channel_se is not None
            else bool(getattr(CFG, "use_channel_se", True))
        )
        self.channel_se_reduction = (
            int(channel_se_reduction)
            if channel_se_reduction is not None
            else int(getattr(CFG, "channel_se_reduction", 8))
        )
        self.use_mixed_pooling = (
            bool(use_mixed_pooling)
            if use_mixed_pooling is not None
            else bool(getattr(CFG, "use_mixed_pooling", True))
        )
        self.freq_branch_channels = (
            int(freq_branch_channels)
            if freq_branch_channels is not None
            else int(getattr(CFG, "freq_branch_channels", self.tcn_channels))
        )
        self.freq_branch_kernel_size = (
            int(freq_branch_kernel_size)
            if freq_branch_kernel_size is not None
            else int(getattr(CFG, "freq_branch_kernel_size", 3))
        )
        self.freq_branch_dropout = (
            float(freq_branch_dropout)
            if freq_branch_dropout is not None
            else float(getattr(CFG, "freq_branch_dropout", self.tcn_dropout))
        )
        self.use_freq_branch_se = (
            bool(use_freq_branch_se)
            if use_freq_branch_se is not None
            else bool(getattr(CFG, "use_freq_branch_se", self.use_channel_se))
        )
        self.freq_branch_se_reduction = (
            int(freq_branch_se_reduction)
            if freq_branch_se_reduction is not None
            else int(getattr(CFG, "freq_branch_se_reduction", self.channel_se_reduction))
        )
        self.fft_mag_log1p = (
            bool(fft_mag_log1p)
            if fft_mag_log1p is not None
            else bool(getattr(CFG, "fft_mag_log1p", True))
        )
        self.use_sensor_group_attention = (
            bool(use_sensor_group_attention)
            if use_sensor_group_attention is not None
            else bool(getattr(CFG, "use_sensor_group_attention", False))
        )
        self.fft_mag_norm = (
            str(fft_mag_norm).strip().lower()
            if fft_mag_norm is not None
            else str(getattr(CFG, "fft_mag_norm", "per_sample_channel")).strip().lower()
        )
        self.use_deformable_tcn = (
            bool(use_deformable_tcn)
            if use_deformable_tcn is not None
            else bool(getattr(CFG, "use_deformable_tcn", False))
        )
        self.deformable_conv_mode = (
            str(deformable_conv_mode).strip().lower()
            if deformable_conv_mode is not None
            else str(getattr(CFG, "deformable_conv_mode", "conv1_only")).strip().lower()
        )
        self.deformable_causal_mode = (
            str(deformable_causal_mode).strip().lower()
            if deformable_causal_mode is not None
            else str(getattr(CFG, "deformable_causal_mode", "strict")).strip().lower()
        )
        self.deform_offset_kernel_size = (
            int(deform_offset_kernel_size)
            if deform_offset_kernel_size is not None
            else int(getattr(CFG, "deform_offset_kernel_size", 3))
        )
        self.deform_offset_l1_weight = (
            float(deform_offset_l1_weight)
            if deform_offset_l1_weight is not None
            else float(getattr(CFG, "deform_offset_l1_weight", 1e-4))
        )
        self.use_hybrid_deform_multiscale_tcn = (
            bool(use_hybrid_deform_multiscale_tcn)
            if use_hybrid_deform_multiscale_tcn is not None
            else bool(getattr(CFG, "use_hybrid_deform_multiscale_tcn", False))
        )
        self.hybrid_tcn_kernel_sizes = (
            tuple(int(k) for k in hybrid_tcn_kernel_sizes)
            if hybrid_tcn_kernel_sizes is not None
            else tuple(int(k) for k in getattr(CFG, "hybrid_tcn_kernel_sizes", (3, 5, 7)))
        )
        self.use_levelwise_tcn_aggregation = (
            bool(use_levelwise_tcn_aggregation)
            if use_levelwise_tcn_aggregation is not None
            else bool(getattr(CFG, "use_levelwise_tcn_aggregation", False))
        )
        self.use_uncertainty_temporal_pooling = (
            bool(use_uncertainty_temporal_pooling)
            if use_uncertainty_temporal_pooling is not None
            else bool(getattr(CFG, "use_uncertainty_temporal_pooling", False))
        )

        if self.input_dim <= 0:
            raise ValueError("input_dim 必须 > 0")
        if self.tcn_channels <= 0:
            raise ValueError("tcn_channels 必须 > 0")
        if self.tcn_num_levels <= 0:
            raise ValueError("tcn_num_levels 必须 > 0")
        if self.freq_branch_channels <= 0:
            raise ValueError("freq_branch_channels 必须 > 0")
        if self.tcn_kernel_size <= 1 or self.tcn_kernel_size % 2 == 0:
            raise ValueError("tcn_kernel_size 必须是大于1的奇数")
        if self.freq_branch_kernel_size <= 1 or self.freq_branch_kernel_size % 2 == 0:
            raise ValueError("freq_branch_kernel_size 必须是大于1的奇数")
        if not (0.0 <= self.tcn_dropout < 1.0):
            raise ValueError("tcn_dropout 必须在 [0,1) 范围")
        if not (0.0 <= self.freq_branch_dropout < 1.0):
            raise ValueError("freq_branch_dropout 必须在 [0,1) 范围")
        if self.use_channel_se and self.channel_se_reduction <= 0:
            raise ValueError("channel_se_reduction 必须 > 0")
        if self.use_freq_branch_se and self.freq_branch_se_reduction <= 0:
            raise ValueError("freq_branch_se_reduction 必须 > 0")
        if self.use_sensor_group_attention and self.input_dim < self.raw_sensor_input_dim:
            raise ValueError(
                "use_sensor_group_attention 开启时 input_dim 必须 >= 原始通道数 "
                f"({self.raw_sensor_input_dim})"
            )
        if self.fft_mag_norm not in {"none", "per_sample_channel", "global"}:
            raise ValueError("fft_mag_norm 必须是 none/per_sample_channel/global")
        if self.deformable_conv_mode not in {"conv1_only", "both"}:
            raise ValueError("deformable_conv_mode 必须是 conv1_only 或 both")
        if self.deformable_causal_mode not in {"strict", "relaxed"}:
            raise ValueError("deformable_causal_mode 必须是 strict 或 relaxed")
        if self.deform_offset_kernel_size <= 1 or self.deform_offset_kernel_size % 2 == 0:
            raise ValueError("deform_offset_kernel_size 必须是大于1的奇数")
        if self.deform_offset_l1_weight < 0.0:
            raise ValueError("deform_offset_l1_weight 必须 >= 0")
        self.hybrid_tcn_kernel_sizes = _to_kernel_tuple(
            self.hybrid_tcn_kernel_sizes,
            "hybrid_tcn_kernel_sizes",
        )
        if self.use_hybrid_deform_multiscale_tcn and not self.use_deformable_tcn:
            raise ValueError("use_hybrid_deform_multiscale_tcn 开启时 use_deformable_tcn 必须为 True")

        if self.use_lwpt_frontend:
            if self.lwpt_num_bands < 1:
                raise ValueError("lwpt_num_bands 必须 >= 1")
            self.lwpt_kernel_sizes = _to_kernel_tuple(self.lwpt_kernel_sizes, "lwpt_kernel_sizes")
            if not (0.0 <= self.lwpt_dropout < 1.0):
                raise ValueError("lwpt_dropout 必须在 [0,1) 范围")
            self.lwpt_frontend = LearnableWaveletPacketFrontend(
                in_channels=self.input_dim,
                num_bands=self.lwpt_num_bands,
                kernel_sizes=self.lwpt_kernel_sizes,
                dropout=self.lwpt_dropout,
            )
        else:
            self.lwpt_frontend = nn.Identity()
        self.sensor_group_attention = (
            SensorGroupAttention(
                in_channels=self.raw_sensor_input_dim,
                channel_names=(
                    self.raw_sensor_channel_names
                    if len(self.raw_sensor_channel_names) == self.raw_sensor_input_dim
                    else None
                ),
            )
            if self.use_sensor_group_attention
            else None
        )

        self.time_input_channels = self.input_dim
        self.freq_input_channels = self.input_dim
        # 保留该字段用于历史日志/兼容逻辑，不再作为主干输入通道。
        self.fused_input_channels = self.input_dim * 2

        layers: List[TemporalBlock] = []
        in_ch = self.time_input_channels
        for level in range(self.tcn_num_levels):
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=self.tcn_channels,
                    kernel_size=self.tcn_kernel_size,
                    dilation=2**level,
                    dropout=self.tcn_dropout,
                    use_channel_se=self.use_channel_se,
                    channel_se_reduction=self.channel_se_reduction,
                    use_multi_scale_conv=False,
                    use_deformable_conv=self.use_deformable_tcn,
                    deformable_conv_mode=self.deformable_conv_mode,
                    deformable_causal_mode=self.deformable_causal_mode,
                    deform_offset_kernel_size=self.deform_offset_kernel_size,
                    use_hybrid_deform_multiscale=self.use_hybrid_deform_multiscale_tcn,
                    hybrid_kernel_sizes=self.hybrid_tcn_kernel_sizes,
                    hybrid_use_pool_branch=True,
                    hybrid_fft_temperature=2.0,
                    hybrid_use_fft_adaptive=True,
                )
            )
            in_ch = self.tcn_channels
        self.network = nn.ModuleList(layers)
        self.levelwise_score = (
            nn.Sequential(
                nn.Linear(self.tcn_channels, max(self.tcn_channels // 4, 1)),
                nn.ReLU(inplace=True),
                nn.Linear(max(self.tcn_channels // 4, 1), 1),
            )
            if self.use_levelwise_tcn_aggregation
            else None
        )
        self.uncertainty_gate = (
            nn.Conv1d(self.tcn_channels, 1, kernel_size=1, bias=True)
            if self.use_uncertainty_temporal_pooling
            else None
        )

        freq_pad = (self.freq_branch_kernel_size - 1) // 2
        self.freq_projection = nn.Sequential(
            nn.Conv1d(
                in_channels=self.freq_input_channels,
                out_channels=self.freq_branch_channels,
                kernel_size=self.freq_branch_kernel_size,
                padding=freq_pad,
                bias=False,
            ),
            nn.BatchNorm1d(self.freq_branch_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.freq_branch_dropout),
            nn.Conv1d(
                in_channels=self.freq_branch_channels,
                out_channels=self.freq_branch_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.freq_branch_channels),
            nn.ReLU(inplace=True),
        )
        self.freq_channel_se = (
            ChannelSE1D(
                channels=self.freq_branch_channels,
                reduction=self.freq_branch_se_reduction,
            )
            if self.use_freq_branch_se
            else None
        )
        pool_dim_factor = 2 if self.use_mixed_pooling else 1
        self.time_pooled_channels = self.tcn_channels * pool_dim_factor
        self.freq_pooled_channels = self.freq_branch_channels * pool_dim_factor
        self.late_fusion_channels = self.time_pooled_channels + self.freq_pooled_channels
        self.late_fusion_proj = nn.Sequential(
            nn.Linear(self.late_fusion_channels, self.tcn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.tcn_dropout),
        )
        self.classifier = nn.Linear(self.tcn_channels, self.num_classes)
        self._latest_deform_offset_l1: torch.Tensor | None = None

    def _build_fft_magnitude(self, raw_input: torch.Tensor) -> torch.Tensor:
        fft_in = raw_input.float()
        fft_mag = torch.abs(torch.fft.fft(fft_in, dim=-1))
        if self.fft_mag_log1p:
            fft_mag = torch.log1p(fft_mag)
        if self.fft_mag_norm == "per_sample_channel":
            denom = fft_mag.amax(dim=-1, keepdim=True).clamp_min(1e-6)
            fft_mag = fft_mag / denom
        elif self.fft_mag_norm == "global":
            denom = fft_mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            fft_mag = fft_mag / denom
        return fft_mag.to(dtype=raw_input.dtype)

    def _forward_tcn_blocks(
        self,
        x_in: torch.Tensor,
        lengths: torch.Tensor,
        collect_aux: bool,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        y = x_in
        level_outputs: List[torch.Tensor] = []
        channel_attn_per_layer: List[torch.Tensor] = []
        hybrid_gate_means: List[torch.Tensor] = []
        deform_offset_means: List[torch.Tensor] = []
        deform_offset_stds: List[torch.Tensor] = []
        deform_offset_max_abs: List[torch.Tensor] = []
        deform_offset_l1 = y.new_zeros(())
        for block in self.network:
            if collect_aux:
                block_out = block(y, return_aux=True)
                y = block_out[0]
                block_aux = block_out[1]
                if "channel_attn_weights" in block_aux:
                    channel_attn_per_layer.append(block_aux["channel_attn_weights"])
                if "deform_offset_mean" in block_aux:
                    deform_offset_means.append(block_aux["deform_offset_mean"])
                    deform_offset_stds.append(block_aux["deform_offset_std"])
                    deform_offset_max_abs.append(block_aux["deform_offset_max_abs"])
                if "hybrid_gate_mean" in block_aux:
                    hybrid_gate_means.append(block_aux["hybrid_gate_mean"])
            else:
                y = block(y)
            deform_offset_l1 = deform_offset_l1 + block.get_deform_offset_l1()
            level_outputs.append(y)

        level_tokens = [feat.transpose(1, 2) for feat in level_outputs]
        level_pooled = [masked_mean_pooling(tokens, lengths=lengths) for tokens in level_tokens]

        levelwise_alpha: torch.Tensor | None = None
        if self.use_levelwise_tcn_aggregation:
            if self.levelwise_score is None:
                raise RuntimeError("levelwise_score 未初始化")
            if len(level_outputs) == 0:
                raise RuntimeError("TCN 层输出为空，无法做层级聚合")
            level_scores = torch.cat([self.levelwise_score(pooled) for pooled in level_pooled], dim=1)
            levelwise_alpha = torch.softmax(level_scores, dim=1)
            fused = level_outputs[0].new_zeros(level_outputs[0].shape)
            for idx, feat in enumerate(level_outputs):
                weight = levelwise_alpha[:, idx].view(-1, 1, 1)
                fused = fused + feat * weight
            y = fused

        self._latest_deform_offset_l1 = deform_offset_l1

        aux: Dict[str, object] = {}
        if collect_aux:
            aux["hybrid_gate_mean"] = (
                float(torch.stack(hybrid_gate_means, dim=0).mean().detach().item())
                if len(hybrid_gate_means) > 0
                else None
            )
            aux["levelwise_alpha"] = (
                levelwise_alpha.detach().cpu() if levelwise_alpha is not None else None
            )
            if len(channel_attn_per_layer) > 0:
                stacked = torch.stack(channel_attn_per_layer, dim=0)
                aux["channel_attn_weights_summary"] = stacked.mean(dim=0).detach().cpu()
            else:
                aux["channel_attn_weights_summary"] = None
            if self.use_deformable_tcn:
                if len(deform_offset_means) > 0:
                    aux["deform_offset_mean"] = float(
                        torch.stack(deform_offset_means, dim=0).mean().detach().item()
                    )
                    aux["deform_offset_std"] = float(
                        torch.stack(deform_offset_stds, dim=0).mean().detach().item()
                    )
                    aux["deform_offset_max_abs"] = float(
                        torch.stack(deform_offset_max_abs, dim=0).amax().detach().item()
                    )
                else:
                    aux["deform_offset_mean"] = 0.0
                    aux["deform_offset_std"] = 0.0
                    aux["deform_offset_max_abs"] = 0.0
                aux["deform_offset_l1"] = float(deform_offset_l1.detach().item())
                aux["deform_offset_l1_weighted"] = float(
                    (deform_offset_l1 * self.deform_offset_l1_weight).detach().item()
                )
        return y, aux

    def _forward_freq_branch(
        self,
        fft_mag: torch.Tensor,
        collect_aux: bool,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        y_freq = self.freq_projection(fft_mag)
        freq_attn_weights = None
        if self.freq_channel_se is not None:
            y_freq, freq_attn_weights = self.freq_channel_se(y_freq)

        aux: Dict[str, object] = {}
        if collect_aux:
            aux["freq_channel_attn_weights_summary"] = (
                freq_attn_weights.detach().cpu()
                if freq_attn_weights is not None
                else None
            )
        return y_freq, aux

    def _pool_branch_tokens(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        pooled_mean = masked_mean_pooling(tokens, lengths=lengths)
        if not self.use_mixed_pooling:
            return pooled_mean
        pooled_max = masked_max_pooling(tokens, lengths=lengths)
        return torch.cat([pooled_mean, pooled_max], dim=1)

    def _pool_time_tokens_with_uncertainty(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.uncertainty_gate is None:
            raise RuntimeError("uncertainty_gate 未初始化")
        token_map = tokens.transpose(1, 2)  # [B, C, T]
        gate_logits = self.uncertainty_gate(token_map).squeeze(1)  # [B, T]
        gate = torch.sigmoid(gate_logits)
        _, seq_len, _ = tokens.shape
        lengths_device = lengths.to(device=tokens.device)
        time_index = torch.arange(seq_len, device=tokens.device).view(1, -1)
        valid_mask = time_index < lengths_device.view(-1, 1)
        gate = gate * valid_mask.to(dtype=gate.dtype)
        denom = gate.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled_weighted = (tokens * gate.unsqueeze(-1)).sum(dim=1) / denom
        if self.use_mixed_pooling:
            pooled_max = masked_max_pooling(tokens, lengths=lengths)
            pooled = torch.cat([pooled_weighted, pooled_max], dim=1)
        else:
            pooled = pooled_weighted
        valid_token_count = valid_mask.sum().clamp_min(1)
        gate_mean = gate.sum() / valid_token_count.to(dtype=gate.dtype)
        return pooled, gate_mean

    def _extract_sequence_features_with_aux(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        collect_aux: bool,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        if x.ndim != 3:
            raise ValueError(f"输入必须是 [B,T,C]，实际={tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"输入通道数不匹配，期望={self.input_dim}，实际={x.shape[-1]}")
        if lengths.ndim != 1:
            raise ValueError(f"lengths 必须是 [B]，实际={tuple(lengths.shape)}")
        if lengths.shape[0] != x.shape[0]:
            raise ValueError(f"lengths 与 batch 不匹配: lengths={lengths.shape[0]} batch={x.shape[0]}")

        raw_input = x.transpose(1, 2)  # [B, C, T]
        if self.sensor_group_attention is not None:
            raw_part = raw_input[:, : self.raw_sensor_input_dim, :]
            enhanced_raw = self.sensor_group_attention(raw_part)
            if self.input_dim > self.raw_sensor_input_dim:
                raw_input = torch.cat(
                    [enhanced_raw, raw_input[:, self.raw_sensor_input_dim :, :]],
                    dim=1,
                )
            else:
                raw_input = enhanced_raw
        time_feat = self.lwpt_frontend(raw_input)  # [B, C, T]
        fft_mag = self._build_fft_magnitude(raw_input)  # [B, C, T]
        y_time, tcn_aux = self._forward_tcn_blocks(
            x_in=time_feat,
            lengths=lengths,
            collect_aux=collect_aux,
        )
        y_freq, freq_aux = self._forward_freq_branch(
            fft_mag=fft_mag,
            collect_aux=collect_aux,
        )

        time_tokens = y_time.transpose(1, 2)  # [B, T, C_tcn]
        freq_tokens = y_freq.transpose(1, 2)  # [B, T, C_freq]
        uncertainty_gate_mean: torch.Tensor | None = None
        if self.use_uncertainty_temporal_pooling:
            pooled_time, uncertainty_gate_mean = self._pool_time_tokens_with_uncertainty(
                tokens=time_tokens,
                lengths=lengths,
            )
        else:
            pooled_time = self._pool_branch_tokens(time_tokens, lengths=lengths)
        pooled_freq = self._pool_branch_tokens(freq_tokens, lengths=lengths)
        late_fused = torch.cat([pooled_time, pooled_freq], dim=1)  # [B, C_tcn + C_freq]
        pooled = self.late_fusion_proj(late_fused)  # [B, C_tcn]

        aux: Dict[str, object] = {}
        if collect_aux:
            aux.update(tcn_aux)
            aux.update(freq_aux)
            aux["fft_shape"] = tuple(fft_mag.shape)
            aux["time_stream_shape"] = tuple(y_time.shape)
            aux["freq_stream_shape"] = tuple(y_freq.shape)
            aux["time_pooled_shape"] = tuple(pooled_time.shape)
            aux["freq_pooled_shape"] = tuple(pooled_freq.shape)
            aux["late_fusion_shape"] = tuple(late_fused.shape)
            aux["use_mixed_pooling"] = self.use_mixed_pooling
            # 保留该字段，兼容历史分析脚本读取。
            aux["fused_input_shape"] = None
            aux["pooled_shape"] = tuple(pooled.shape)
            aux["tf_seq_shape"] = None
            aux["branch_weights"] = None
            aux["sensor_group_attention_enabled"] = bool(self.sensor_group_attention is not None)
            aux["hybrid_gate_mean"] = tcn_aux.get("hybrid_gate_mean")
            aux["levelwise_alpha"] = tcn_aux.get("levelwise_alpha")
            aux["uncertainty_gate_mean"] = (
                float(uncertainty_gate_mean.detach().item())
                if uncertainty_gate_mean is not None
                else None
            )
        return pooled, aux

    def _extract_sequence_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        pooled, _ = self._extract_sequence_features_with_aux(
            x=x,
            lengths=lengths,
            collect_aux=False,
        )
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        pooled, _ = self._extract_sequence_features_with_aux(
            x=x,
            lengths=lengths,
            collect_aux=False,
        )
        logits = self.classifier(pooled)
        return logits

    def forward_with_aux(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        pooled, aux = self._extract_sequence_features_with_aux(
            x=x,
            lengths=lengths,
            collect_aux=True,
        )
        logits = self.classifier(pooled)
        aux.setdefault("hybrid_gate_mean", None)
        aux.setdefault("levelwise_alpha", None)
        aux.setdefault("uncertainty_gate_mean", None)
        # 供辅助特征损失（如 SupCon/CenterLoss）复用，避免二次前向提特征。
        aux["pooled_features"] = pooled
        aux["logits_shape"] = tuple(logits.shape)
        return logits, aux

    def freeze_feature_extractor(self, freeze: bool = True) -> None:
        modules: List[nn.Module] = [
            self.lwpt_frontend,
            self.network,
            self.freq_projection,
            self.late_fusion_proj,
        ]
        if self.sensor_group_attention is not None:
            modules.append(self.sensor_group_attention)
        if self.levelwise_score is not None:
            modules.append(self.levelwise_score)
        if self.uncertainty_gate is not None:
            modules.append(self.uncertainty_gate)
        if self.freq_channel_se is not None:
            modules.append(self.freq_channel_se)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = not freeze

    def freeze_frontend(self, freeze: bool = True) -> None:
        """冻结前端层（LWPT + freq_projection），保留 TCN backbone 可训练。

        用途：当输入含交叉传感器残差通道时，前端层已在源域学到
        域不变的残差频率分解模式。冻结这些层可防止目标域少量数据
        破坏源域学到的残差表征。
        """
        modules: List[nn.Module] = [
            self.lwpt_frontend,
            self.freq_projection,
        ]
        if self.sensor_group_attention is not None:
            modules.append(self.sensor_group_attention)
        if self.freq_channel_se is not None:
            modules.append(self.freq_channel_se)
        frozen_count = 0
        for module in modules:
            for param in module.parameters():
                param.requires_grad = not freeze
                frozen_count += 1
        if freeze:
            print(f"[freeze_frontend] frozen {frozen_count} parameters in lwpt_frontend + freq_projection")

    def get_additional_regularization_loss(self) -> torch.Tensor:
        if not self.use_deformable_tcn or self.deform_offset_l1_weight <= 0.0:
            return self.classifier.weight.new_zeros(())
        if self._latest_deform_offset_l1 is None:
            return self.classifier.weight.new_zeros(())
        if not torch.isfinite(self._latest_deform_offset_l1):
            return self.classifier.weight.new_zeros(())
        return self._latest_deform_offset_l1 * self.deform_offset_l1_weight

    def get_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self._extract_sequence_features(x=x, lengths=lengths)
