"""
visualization.py

迁移学习可视化模块

职责:
1. 双域混淆矩阵: 源域和目标域并排显示
2. t-SNE 特征可视化: 源域和目标域特征在同一图中展示
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _setup_matplotlib():
    """延迟导入并配置 matplotlib，使用 Agg 后端。"""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _get_chinese_font() -> Optional[str]:
    """
    获取可用的中文字体。

    回退顺序: Microsoft YaHei → SimHei → Noto Sans CJK SC → DejaVu Sans
    """
    plt = _setup_matplotlib()
    import matplotlib.font_manager as fm

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    return None


def save_dual_confusion_matrix(
    cm_source: np.ndarray,
    cm_target: np.ndarray,
    output_path: Path,
    class_names: Sequence[str],
    normalize: bool = True,
    source_name: str = "Source",
    target_name: str = "Target",
) -> Optional[Path]:
    """
    将源域和目标域混淆矩阵并排保存为 PNG。

    参数:
        cm_source: 源域混淆矩阵 [N, N]
        cm_target: 目标域混淆矩阵 [N, N]
        output_path: 输出 PNG 路径
        class_names: 类别名称列表
        normalize: 是否按行归一化显示比例
        source_name: 源域标签
        target_name: 目标域标签

    返回:
        成功返回 output_path，失败返回 None
    """
    if cm_source.size == 0 or cm_target.size == 0:
        return None

    try:
        plt = _setup_matplotlib()
    except Exception as exc:
        print(f"[可视化] 跳过双域混淆矩阵导出: matplotlib 不可用 ({exc})")
        return None

    if len(class_names) != cm_source.shape[0]:
        raise ValueError(
            f"class_names 长度({len(class_names)})必须等于混淆矩阵行数({cm_source.shape[0]})"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 归一化处理
    if normalize:
        eps = 1e-8
        cm_src_display = cm_source.astype("float") / (cm_source.sum(axis=1, keepdims=True) + eps)
        cm_tgt_display = cm_target.astype("float") / (cm_target.sum(axis=1, keepdims=True) + eps)
        fmt = ".2f"
        vmax = 1.0
        cbar_label = "Ratio"
    else:
        cm_src_display = cm_source
        cm_tgt_display = cm_target
        fmt = "d"
        vmax = max(cm_source.max(), cm_target.max())
        cbar_label = "Count"

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 统一配色
    cmap = plt.cm.Blues

    # 绘制源域混淆矩阵
    im0 = axes[0].imshow(cm_src_display, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    axes[0].set_xticks(np.arange(len(class_names)))
    axes[0].set_yticks(np.arange(len(class_names)))
    axes[0].set_xticklabels(class_names)
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_title(f"Confusion Matrix ({source_name})")
    axes[0].tick_params(axis="x", labelrotation=45)

    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.ax.set_ylabel(cbar_label, rotation=270, labelpad=14)

    # 标注源域数值
    threshold = float(vmax * 0.5)
    for i in range(cm_src_display.shape[0]):
        for j in range(cm_src_display.shape[1]):
            val = cm_src_display[i, j]
            text = f"{val:{fmt}}" if normalize else str(int(val))
            color = "white" if val > threshold else "black"
            axes[0].text(j, i, text, ha="center", va="center", color=color, fontsize=7)

    # 绘制目标域混淆矩阵
    im1 = axes[1].imshow(cm_tgt_display, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    axes[1].set_xticks(np.arange(len(class_names)))
    axes[1].set_yticks(np.arange(len(class_names)))
    axes[1].set_xticklabels(class_names)
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_title(f"Confusion Matrix ({target_name})")
    axes[1].tick_params(axis="x", labelrotation=45)

    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.ax.set_ylabel(cbar_label, rotation=270, labelpad=14)

    # 标注目标域数值
    for i in range(cm_tgt_display.shape[0]):
        for j in range(cm_tgt_display.shape[1]):
            val = cm_tgt_display[i, j]
            text = f"{val:{fmt}}" if normalize else str(int(val))
            color = "white" if val > threshold else "black"
            axes[1].text(j, i, text, ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def _sample_per_class(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每个类别进行均衡采样。

    参数:
        features: 特征矩阵 [N, D]
        labels: 标签数组 [N]
        max_samples: 每类最大采样数

    返回:
        采样后的 (features, labels)
    """
    if max_samples <= 0:
        return features, labels

    unique_labels = np.unique(labels)
    sampled_features = []
    sampled_labels = []

    rng = np.random.default_rng(seed=42)

    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0]

        if len(indices) > max_samples:
            indices = rng.choice(indices, size=max_samples, replace=False)

        sampled_features.append(features[indices])
        sampled_labels.append(labels[indices])

    return np.vstack(sampled_features), np.concatenate(sampled_labels)


def save_domain_class_tsne(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    max_samples_per_class: int = 200,
    perplexity: float = 30.0,
    source_name: str = "Source",
    target_name: str = "Target",
) -> Optional[Path]:
    """
    生成源域和目标域特征的 t-SNE 可视化。

    颜色 = 真实标签，形状 = 域（源域圆点，目标域三角）

    参数:
        source_features: 源域特征 [N_s, D]
        source_labels: 源域真标签 [N_s]
        target_features: 目标域特征 [N_t, D]
        target_labels: 目标域真标签 [N_t]
        class_names: 类别名称列表
        output_path: 输出 PNG 路径
        max_samples_per_class: 每类采样数限制
        perplexity: t-SNE perplexity 参数
        source_name: 源域标签
        target_name: 目标域标签

    返回:
        成功返回 output_path，失败返回 None
    """
    if source_features.size == 0 or target_features.size == 0:
        return None

    try:
        plt = _setup_matplotlib()
        from sklearn.manifold import TSNE
    except Exception as exc:
        print(f"[可视化] 跳过 t-SNE 导出: 依赖不可用 ({exc})")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 采样以减少计算量
    src_feat, src_lbl = _sample_per_class(source_features, source_labels, max_samples_per_class)
    tgt_feat, tgt_lbl = _sample_per_class(target_features, target_labels, max_samples_per_class)

    # 合并特征
    all_features = np.vstack([src_feat, tgt_feat])
    all_labels = np.concatenate([src_lbl, tgt_lbl])
    domains = np.array([source_name] * len(src_feat) + [target_name] * len(tgt_feat))

    # t-SNE 降维
    n_samples = len(all_features)
    effective_perplexity = min(perplexity, max(5.0, (n_samples - 1) / 3))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        random_state=42,
        learning_rate="auto",
        max_iter=1000,
    )
    coords = tsne.fit_transform(all_features)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))

    # 获取唯一标签并创建颜色映射
    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_classes, 10))
    colors = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}

    # 形状映射
    markers = {source_name: "o", target_name: "^"}

    # 按域和类别绘制散点
    for domain in [source_name, target_name]:
        for label in unique_labels:
            mask = (domains == domain) & (all_labels == label)
            if mask.sum() == 0:
                continue

            class_idx = int(label)
            class_name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)

            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[colors[label]],
                marker=markers[domain],
                label=f"{domain}-{class_name}",
                alpha=0.7,
                s=50,
                edgecolors="white",
                linewidths=0.5,
            )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE Feature Visualization ({source_name} vs {target_name})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
