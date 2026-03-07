"""
fine_tune_metrics.py

不依赖 sklearn/torch 的微调验证指标计算。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    labels = np.unique(np.concatenate([y_true_arr, y_pred_arr], axis=0))
    if labels.size == 0:
        return 0.0

    f1_values: List[float] = []
    for cls in labels.tolist():
        tp = int(np.logical_and(y_true_arr == cls, y_pred_arr == cls).sum())
        fp = int(np.logical_and(y_true_arr != cls, y_pred_arr == cls).sum())
        fn = int(np.logical_and(y_true_arr == cls, y_pred_arr != cls).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if precision + recall == 0.0:
            f1_values.append(0.0)
        else:
            f1_values.append(float((2.0 * precision * recall) / (precision + recall)))
    return float(np.mean(np.asarray(f1_values, dtype=np.float64)))


def _gmean(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true_arr.size == 0:
        return 0.0

    labels = np.unique(y_true_arr)
    recalls = []
    for cls in labels.tolist():
        tp = int(np.logical_and(y_true_arr == cls, y_pred_arr == cls).sum())
        fn = int(np.logical_and(y_true_arr == cls, y_pred_arr != cls).sum())
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    if len(recalls) == 0:
        return 0.0

    eps = 1e-12
    recalls_arr = np.clip(np.asarray(recalls, dtype=np.float64), 0.0, 1.0)
    gmean = float(np.power(np.prod(recalls_arr + eps), 1.0 / recalls_arr.size))
    return min(gmean, 1.0)


def compute_validation_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "val_macro_f1": _macro_f1(y_true, y_pred),
        "val_gmean": _gmean(y_true, y_pred),
    }


def compute_early_stopping_score(metrics: Dict[str, float], metric_name: str) -> float:
    normalized = str(metric_name).strip().lower()
    macro_f1 = float(metrics.get("val_macro_f1", 0.0))
    gmean = float(metrics.get("val_gmean", 0.0))
    if normalized == "macro_f1":
        return macro_f1
    if normalized == "macro_f1_gmean_composite":
        return float((macro_f1 + gmean) * 0.5)
    raise ValueError(
        f"unsupported early stopping metric: {metric_name}. "
        "expected macro_f1 or macro_f1_gmean_composite"
    )
