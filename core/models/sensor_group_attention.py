"""
sensor_group_attention.py

Sensor-group attention module for raw sensor channels.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from essence_forge.core.channel_layout import (
    CROSS_SENSOR_RESIDUAL_CHANNEL_NAMES,
    LEGACY_CORE_CHANNEL_NAMES,
    build_raw_plus_residual_sensor_groups,
    build_raw_sensor_groups,
)


class SensorGroupAttention(nn.Module):
    """Sensor-group attention for legacy presets and dynamic channel layouts."""

    BASE_SENSOR_GROUPS: Dict[str, Tuple[int, ...]] = build_raw_sensor_groups(
        LEGACY_CORE_CHANNEL_NAMES
    )
    RESIDUAL_SENSOR_GROUPS: Dict[str, Tuple[int, ...]] = {
        name: indices
        for name, indices in build_raw_plus_residual_sensor_groups(
            LEGACY_CORE_CHANNEL_NAMES + CROSS_SENSOR_RESIDUAL_CHANNEL_NAMES
        ).items()
        if name.startswith("res_")
    }

    def __init__(
        self,
        in_channels: int = 19,
        channel_names: Sequence[str] | None = None,
        sensor_groups: Mapping[str, Sequence[int]] | None = None,
        embed_dim: int = 32,
        num_heads: int = 4,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_channels_int = int(in_channels)
        resolved_sensor_groups = self._resolve_sensor_groups(
            in_channels=in_channels_int,
            channel_names=channel_names,
            sensor_groups=sensor_groups,
        )
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if conv_kernel_size <= 1 or conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be an odd integer > 1")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        self.in_channels = in_channels_int
        self.sensor_groups = resolved_sensor_groups
        self.group_names = tuple(self.sensor_groups.keys())
        self.num_groups = len(self.group_names)

        pad = conv_kernel_size // 2
        self.group_encoders = nn.ModuleDict()
        self.group_token_proj = nn.ModuleDict()
        self.group_gate_proj = nn.ModuleDict()

        for name in self.group_names:
            group_size = len(self.sensor_groups[name])
            self.group_encoders[name] = nn.Sequential(
                nn.Conv1d(
                    in_channels=group_size,
                    out_channels=group_size,
                    kernel_size=conv_kernel_size,
                    padding=pad,
                    bias=False,
                ),
                nn.BatchNorm1d(group_size),
                nn.ReLU(inplace=True),
            )
            self.group_token_proj[name] = nn.Linear(group_size, embed_dim)
            self.group_gate_proj[name] = nn.Linear(embed_dim, group_size)

        self.cross_group_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @classmethod
    def _normalize_sensor_groups(
        cls,
        sensor_groups: Mapping[str, Sequence[int]],
        in_channels: int,
    ) -> Dict[str, Tuple[int, ...]]:
        normalized: Dict[str, Tuple[int, ...]] = {}
        used_indices: set[int] = set()

        for group_name, group_indices in sensor_groups.items():
            indices = tuple(int(idx) for idx in group_indices)
            if len(indices) == 0:
                raise ValueError(f"Sensor group `{group_name}` must not be empty.")
            for idx in indices:
                if idx < 0 or idx >= int(in_channels):
                    raise ValueError(
                        f"Sensor group `{group_name}` contains out-of-range index {idx} "
                        f"for in_channels={in_channels}."
                    )
            overlap = used_indices.intersection(indices)
            if overlap:
                raise ValueError(
                    f"Sensor group `{group_name}` overlaps indices {sorted(overlap)}."
                )
            normalized[str(group_name)] = indices
            used_indices.update(indices)

        if len(normalized) == 0:
            raise ValueError("SensorGroupAttention requires at least one sensor group.")
        return normalized

    @classmethod
    def _resolve_sensor_groups(
        cls,
        in_channels: int,
        channel_names: Sequence[str] | None,
        sensor_groups: Mapping[str, Sequence[int]] | None,
    ) -> Dict[str, Tuple[int, ...]]:
        if sensor_groups is not None:
            return cls._normalize_sensor_groups(sensor_groups, in_channels)

        if channel_names is not None:
            normalized_channel_names = tuple(str(name) for name in channel_names)
            if len(normalized_channel_names) != int(in_channels):
                raise ValueError(
                    "channel_names length must match in_channels; "
                    f"len(channel_names)={len(normalized_channel_names)}, in_channels={in_channels}"
                )
            dynamic_groups = build_raw_plus_residual_sensor_groups(
                normalized_channel_names
            )
            if len(dynamic_groups) > 0:
                return cls._normalize_sensor_groups(dynamic_groups, in_channels)

        if int(in_channels) == 19:
            return dict(cls.BASE_SENSOR_GROUPS)
        if int(in_channels) == 28:
            resolved = dict(cls.BASE_SENSOR_GROUPS)
            resolved.update(cls.RESIDUAL_SENSOR_GROUPS)
            return resolved
        raise ValueError(
            "SensorGroupAttention could not resolve sensor groups. "
            "Provide sensor_groups, channel_names, or use a legacy preset (19 or 28)."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            [B, C, T]
        """
        if x.ndim != 3:
            raise ValueError(f"Input must be [B, C, T], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.in_channels:
            raise ValueError(
                f"Expected input channels {self.in_channels}, got {x.shape[1]}"
            )

        input_dtype = x.dtype
        device_type = x.device.type

        with torch.autocast(device_type=device_type, enabled=False):
            x_fp32 = x.float()
            encoded_groups: Dict[str, torch.Tensor] = {}
            tokens = []
            for name in self.group_names:
                idx = self.sensor_groups[name]
                group_x = x_fp32[:, idx, :]
                group_feat = self.group_encoders[name](group_x)
                encoded_groups[name] = group_feat

                pooled = group_feat.mean(dim=-1)
                token = self.group_token_proj[name](pooled)
                tokens.append(token)

            group_tokens = torch.stack(tokens, dim=1)  # [B, G, E]
            attended, _ = self.cross_group_attn(group_tokens, group_tokens, group_tokens)
            attended = self.dropout(attended)
            if not torch.isfinite(attended).all():
                attended = torch.nan_to_num(attended, nan=0.0, posinf=0.0, neginf=0.0)

            enhanced = x_fp32.new_zeros(x_fp32.shape)
            for group_idx, name in enumerate(self.group_names):
                attn_token = attended[:, group_idx, :]
                gate = torch.sigmoid(self.group_gate_proj[name](attn_token)).unsqueeze(-1)
                group_feat = encoded_groups[name] * gate
                if not torch.isfinite(group_feat).all():
                    group_feat = torch.nan_to_num(
                        group_feat,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                enhanced[:, self.sensor_groups[name], :] = group_feat

            out = x_fp32 + self.alpha.float() * enhanced
            if not torch.isfinite(out).all():
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return out.to(dtype=input_dtype)
