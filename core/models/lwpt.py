"""
lwpt.py

LWPT 风格可学习频带前端。

设计目标：
1. 使用每通道 depthwise filterbank 展开多频带特征。
2. 通过样本自适应频带注意力完成加权融合。
3. 用 1x1 投影回输入通道并做残差连接，保证输出形状与输入一致。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_kernel_sizes(num_bands: int, kernel_sizes: Sequence[int]) -> Tuple[int, ...]:
    """
    将 kernel 配置对齐到 `num_bands`。

    规则：
    - 如果输入长度小于 `num_bands`，循环补齐。
    - 如果输入长度大于 `num_bands`，截断。
    - 每个 kernel 必须是大于 1 的奇数，确保时序长度不变。
    """

    if num_bands <= 0:
        raise ValueError("num_bands 必须 > 0")
    if len(kernel_sizes) == 0:
        raise ValueError("kernel_sizes 不能为空")

    raw = [int(k) for k in kernel_sizes]
    for k in raw:
        if k <= 1 or k % 2 == 0:
            raise ValueError(f"kernel_size 必须是大于 1 的奇数，当前 {k}")

    out: list[int] = []
    idx = 0
    while len(out) < num_bands:
        out.append(raw[idx % len(raw)])
        idx += 1
    return tuple(out[:num_bands])


class LearnableWaveletPacketFrontend(nn.Module):
    """
    可学习频带前端（LWPT 风格）。

    维度路径：
    - 输入: `x` `[B, C, T]`
    - 频带展开: `x_bands` `[B, C*K, T]`
    - 频带加权后回投影: `x_proj` `[B, C, T]`
    - 残差输出: `y` `[B, C, T]`
    """

    def __init__(
        self,
        in_channels: int,
        num_bands: int,
        kernel_sizes: Sequence[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels 必须 > 0")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError("dropout 必须在 [0,1) 范围")

        self.in_channels = int(in_channels)
        self.num_bands = int(num_bands)
        self.kernel_sizes = _normalize_kernel_sizes(self.num_bands, kernel_sizes)
        self.dropout = nn.Dropout(float(dropout))

        # 每个频带一个 depthwise 时序卷积：`[B,C,T] -> [B,C,T]`。
        self.band_filters = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=self.in_channels,
                    bias=False,
                )
                for k in self.kernel_sizes
            ]
        )

        # 用样本级频带统计量生成频带权重（Softmax 后和为 1）。
        self.band_attn = nn.Sequential(
            nn.Linear(self.num_bands, self.num_bands),
            nn.Tanh(),
            nn.Linear(self.num_bands, self.num_bands),
        )

        # `C*K -> C` 的 1x1 回投影。
        self.project = nn.Conv1d(self.in_channels * self.num_bands, self.in_channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入:
        - `x`: `[B, C, T]`
        输出:
        - `y`: `[B, C, T]`
        """

        if x.ndim != 3:
            raise ValueError(f"LWPT 输入必须是 [B,C,T]，当前 {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"LWPT 输入通道不匹配，期望 {self.in_channels}，实际 {x.shape[1]}"
            )

        # 1) 多频带展开。
        band_feats = [F.relu(conv(x), inplace=False) for conv in self.band_filters]
        x_bands = torch.cat(band_feats, dim=1)  # [B, C*K, T]

        # 2) 频带注意力（样本自适应）。
        # 先转为 [B, K, C, T]，再汇聚得到 [B, K] 作为频带统计。
        B, _, T = x_bands.shape
        band_view = x_bands.view(B, self.num_bands, self.in_channels, T)
        band_summary = band_view.mean(dim=(2, 3))  # [B, K]
        band_logits = self.band_attn(band_summary)  # [B, K]
        band_weights = torch.softmax(band_logits, dim=-1).view(B, self.num_bands, 1, 1)

        weighted = band_view * band_weights
        weighted_flat = weighted.view(B, self.in_channels * self.num_bands, T)  # [B, C*K, T]

        # 3) 回投影 + 残差。
        proj = self.project(weighted_flat)  # [B, C, T]
        proj = self.norm(proj)
        proj = self.dropout(proj)
        y = x + proj
        return y

