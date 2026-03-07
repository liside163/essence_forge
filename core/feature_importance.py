"""
feature_importance.py

特征重要性分析（基于特征扰动方法）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

from essence_forge.core.runtime_config import CFG
from essence_forge.core.channel_layout import build_input_feature_names, channel_names_from_specs
from essence_forge.core.datasets import DeterministicWindowDataset, collate_padded
from essence_forge.core.model_checkpoint import load_temporal_convnet_from_checkpoint
from essence_forge.core.rflymad_io import MissionLoader
from essence_forge.core.utils import read_json, resolve_device


def build_feature_names() -> tuple[str, ...]:
    raw_channel_names = channel_names_from_specs(getattr(CFG, "channels", ()))
    residual_channel_names = (
        tuple(str(name) for name in getattr(CFG, "cross_sensor_residual_channel_names", ()))
        if bool(getattr(CFG, "use_cross_sensor_residuals", False))
        else ()
    )
    return build_input_feature_names(
        raw_channel_names=raw_channel_names,
        residual_channel_names=residual_channel_names,
        include_health_mask=bool(getattr(CFG, "concat_health_mask_channels", True)),
    )


def analyze_feature_importance(
    run_dir: Path,
    checkpoint_path: str,
    output_dir: Path,
    num_samples: int = 200,
) -> Dict:
    """
    分析模型特征重要性（基于特征扰动方法）

    原理：逐个通道添加噪声，观察模型预测变化。
    变化越大，说明该通道越重要。

    参数:
        run_dir: 运行目录
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录
        num_samples: 用于分析的样本数

    返回:
        特征重要性结果字典
    """
    device = resolve_device(CFG.device)

    # 加载模型
    print(f"[特征分析] 加载模型: {checkpoint_path}")
    model, _, load_meta = load_temporal_convnet_from_checkpoint(
        checkpoint=checkpoint_path,
        device=device,
    )
    if load_meta["model_kwargs_source"] == "inferred_from_state_dict":
        print(
            "[WARN] checkpoint 缺少 model_init_kwargs，已基于 state_dict 推断结构加载。"
        )
    model.eval()

    # 加载数据
    test_records = read_json(run_dir / "split_source_test.json")
    if not test_records:
        test_records = read_json(run_dir / "split_source_val.json")

    stats = read_json(run_dir / "source_stats.json")
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    loader = MissionLoader(max_cache_items=128)
    test_dataset = DeterministicWindowDataset(
        records=test_records[:num_samples],
        zscore_mean=mean,
        zscore_std=std,
        loader=loader,
        windows_per_scale=1,
        base_seed=42,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_padded,
    )

    # 收集数据
    print("[特征分析] 收集数据...")
    all_x = []
    all_lengths = []

    for x, lengths, _ in test_loader:
        all_x.append(x.numpy())
        all_lengths.append(lengths.numpy())

    X = np.concatenate(all_x, axis=0)  # [N, T, C]
    lengths = np.concatenate(all_lengths, axis=0)

    n_samples = min(200, len(X))
    X = X[:n_samples]
    lengths = lengths[:n_samples]

    # 获取通道数
    n_channels = X.shape[2]

    # 通道名称
    channel_names = list(build_feature_names())

    # 确保通道名称数量匹配
    if len(channel_names) > n_channels:
        channel_names = channel_names[:n_channels]
    elif len(channel_names) < n_channels:
        channel_names.extend([f"ch_{i}" for i in range(len(channel_names), n_channels)])

    # 计算基准预测
    print("[特征分析] 计算基准预测...")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(X_tensor, lengths_tensor)
        base_probs = torch.softmax(logits, dim=-1).cpu().numpy()
        base_preds = base_probs.argmax(axis=1)

    # 计算每个通道的重要性
    print("[特征分析] 计算特征重要性（扰动法）...")
    channel_importance = np.zeros(n_channels)
    noise_std = 0.5  # 扰动强度

    for ch in range(n_channels):
        X_perturbed = X.copy()
        # 对该通道添加高斯噪声
        X_perturbed[:, :, ch] += np.random.normal(0, noise_std, X_perturbed[:, :, ch].shape)

        X_pert_tensor = torch.tensor(X_perturbed, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits_pert = model(X_pert_tensor, lengths_tensor)
            pert_probs = torch.softmax(logits_pert, dim=-1).cpu().numpy()

        # 计算预测概率变化（KL 散度或 L2 距离）
        # 使用概率向量的 L2 距离
        diff = np.mean(np.sum((base_probs - pert_probs) ** 2, axis=1))
        channel_importance[ch] = diff

        if (ch + 1) % 5 == 0:
            print(f"  已处理 {ch + 1}/{n_channels} 通道")

    # 排序
    sorted_idx = np.argsort(channel_importance)[::-1]

    # 输出结果
    print("\n=== 特征重要性排名 ===")
    for i, idx in enumerate(sorted_idx):
        print(f"{i+1}. {channel_names[idx]}: {channel_importance[idx]:.6f}")

    # 可视化
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    top_k = min(20, len(sorted_idx))
    top_idx = sorted_idx[:top_k]

    plt.barh(range(top_k), channel_importance[top_idx][::-1])
    plt.yticks(
        range(top_k),
        [channel_names[i] for i in top_idx[::-1]]
    )
    plt.xlabel("Importance (Prediction Change)")
    plt.title("Top 20 Feature Importance (Perturbation Method)")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()

    # 保存结果
    results = {
        "method": "perturbation",
        "noise_std": noise_std,
        "n_samples": n_samples,
        "channel_importance": {
            channel_names[i]: float(channel_importance[i])
            for i in range(len(channel_importance))
        },
        "sorted_channels": [
            channel_names[i] for i in sorted_idx
        ],
        "top_10": [
            {"rank": i + 1, "channel": channel_names[idx], "importance": float(channel_importance[idx])}
            for i, idx in enumerate(sorted_idx[:10])
        ],
    }

    with open(output_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[特征分析] 结果已保存到 {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="特征重要性分析")
    parser.add_argument("--run-dir", type=str, required=True, help="运行目录")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./feature_analysis",
        help="输出目录"
    )
    parser.add_argument(
        "--num-samples", type=int, default=200,
        help="用于分析的样本数"
    )
    args = parser.parse_args()

    analyze_feature_importance(
        run_dir=Path(args.run_dir),
        checkpoint_path=args.checkpoint,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
    )
