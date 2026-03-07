"""
masking.py

变长序列掩码池化实现。

包含:
1. masked_mean_pooling: 忽略 padding 的均值池化。
2. masked_max_pooling: 忽略 padding 的最大池化。
"""

from __future__ import annotations

import torch


def _build_mask(
    lengths: torch.Tensor,
    time_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    根据 lengths 构建 [B, T] 有效位置掩码。
    """
    arange = torch.arange(time_steps, device=device).unsqueeze(0)  # [1, T]
    lengths_expanded = lengths.unsqueeze(1)  # [B, 1]
    return arange < lengths_expanded


def masked_mean_pooling(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    对变长序列做 Masked Mean Pooling
    
    参数:
        x: [B, T, C] 特征序列
        lengths: [B] 每个样本的有效长度
        
    返回:
        pooled: [B, C] 聚合后的特征
        
    示例:
        >>> x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, channels=64
        >>> lengths = torch.tensor([8, 6])  # 有效长度分别为8和6
        >>> pooled = masked_mean_pooling(x, lengths)  # [2, 64]
    """
    B, T, C = x.shape
    device = x.device

    # 构建 mask: [B, T], mask[b, t]=True 表示该时间步有效。
    mask = _build_mask(lengths=lengths, time_steps=T, device=device)
    
    # 扩展 mask 到 [B, T, C] 用于掩码计算
    mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
    
    # 对有效位置求和并除以有效长度
    # sum: [B, C]
    sum_features = (x * mask_expanded).sum(dim=1)  # [B, C]
    
    # 防止除零
    lengths_clamped = lengths.clamp(min=1).float().unsqueeze(-1)  # [B, 1]
    
    pooled = sum_features / lengths_clamped  # [B, C]

    return pooled


def masked_max_pooling(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    对变长序列做 Masked Max Pooling。

    参数:
        x: [B, T, C] 特征序列
        lengths: [B] 每个样本的有效长度

    返回:
        pooled: [B, C] 聚合后的特征
    """
    B, T, _ = x.shape
    device = x.device
    mask = _build_mask(lengths=lengths, time_steps=T, device=device)  # [B, T]

    # 将 padding 位置填充为极小值，避免参与 max。
    fill_value = torch.finfo(x.dtype).min
    x_masked = x.masked_fill(~mask.unsqueeze(-1), fill_value)
    pooled = x_masked.max(dim=1).values  # [B, C]

    # 对长度为 0 的样本输出 0，避免保留 fill_value。
    valid = (lengths > 0).unsqueeze(-1)  # [B, 1]
    pooled = torch.where(valid, pooled, torch.zeros_like(pooled))

    return pooled
