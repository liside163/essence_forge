"""
metrics.py

评估指标计算模块
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 11
) -> Dict[str, float]:
    """
    计算分类评估指标
    
    参数:
        y_true: 真实标签 [N]
        y_pred: 预测标签 [N]
        num_classes: 类别数
        
    返回:
        包含各指标的字典
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算 G-mean（几何均值召回率）
    
    G-mean 对类别不平衡更敏感，是 ACS-ATCN 论文使用的优化目标
    
    公式:
        G-mean = (∏ Recall_i)^(1/n_classes)
        
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        G-mean 分数 (0 到 1)
    """
    # -----------------------------
    # 关键点：
    # - 不能把 recall=0 的类别过滤掉再算 G-mean，否则会出现“某个小类碰巧 recall=1 -> G-mean≈1”的虚高现象；
    # - 默认只在 y_true 实际出现的类别上计算（验证集中没出现的类无法评价），但这些出现类的 recall=0 必须保留。
    # -----------------------------
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return 0.0

    labels_present = np.unique(y_true)
    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels_present.tolist(),
        average=None,
        zero_division=0,
    ).astype(np.float64)
    
    # 过滤掉召回率为 0 或未出现的类别
    # 保留 recall=0 的类别（这是 G-mean 惩罚“漏检类”的核心所在）
    valid_recalls = recalls
    
    if len(valid_recalls) == 0:
        return 0.0
    
    # 计算几何平均
    # 使用 eps 避免 prod=0 导致梯度/日志完全塌陷（同时保持对 recall=0 的强惩罚）
    eps = 1e-12
    valid_recalls = np.clip(valid_recalls, 0.0, 1.0)
    gmean = np.power(np.prod(valid_recalls + eps), 1.0 / len(valid_recalls))
    # 数值稳定：recall≈1 时 valid_recalls+eps 可能让 gmean 略微 >1
    if gmean > 1.0:
        gmean = 1.0
    
    return float(gmean)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 11
) -> np.ndarray:
    """
    计算混淆矩阵
    
    返回:
        cm: [num_classes, num_classes] 混淆矩阵
        cm[i, j] = 真实类别 i 被预测为类别 j 的样本数
    """
    return confusion_matrix(
        y_true, y_pred,
        labels=list(range(num_classes))
    )
