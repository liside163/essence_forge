"""
evaluate.py

模型评估模块

职责:
1. 在测试集上评估模型
2. 计算各项指标
3. 生成混淆矩阵
4. 导出混淆矩阵热力图（PNG）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

import sys

from essence_forge.core.runtime_config import CFG
from essence_forge.core.datasets import DeterministicWindowDataset, collate_padded
from essence_forge.core.rflymad_io import MissionLoader
from essence_forge.core.model_checkpoint import load_temporal_convnet_from_checkpoint
from essence_forge.core.metrics import compute_metrics, compute_gmean, compute_confusion_matrix
from essence_forge.core.utils import read_json, write_json, resolve_device, resolve_num_workers
from essence_forge.core.visualization import save_dual_confusion_matrix, save_domain_class_tsne


@dataclass
class EvalResult:
    """评估结果"""
    model_name: str
    domain: str  # "source" or "target"
    split: str
    accuracy: float
    macro_f1: float
    macro_recall: float
    weighted_f1: float
    gmean: float
    confusion_matrix: List[List[int]]
    per_class_metrics: List[Dict[str, Any]]
    top_confusion_pairs: List[Dict[str, Any]]
    attention_summary: Dict[str, Any]
    global_threshold: Optional[float] = None
    macro_f1_before_threshold: Optional[float] = None
    threshold_mode: Optional[str] = None
    per_class_thresholds: Optional[List[float]] = None
    threshold_applied: Optional[bool] = None
    threshold_selection_reason: Optional[str] = None
    probability_temperature: Optional[float] = None
    nofault_logit_bias: Optional[float] = None


THRESHOLD_OBJECTIVE_ACC_FIRST = "accuracy_first_lexicographic"
THRESHOLD_OBJECTIVE_F1_WITH_ACC_FLOOR = "macro_f1_with_accuracy_floor"


def _normalize_threshold_search_objective(objective: Optional[str]) -> str:
    raw = str(objective or THRESHOLD_OBJECTIVE_ACC_FIRST).strip().lower()
    if raw in {
        THRESHOLD_OBJECTIVE_ACC_FIRST,
        "accuracy_first",
        "accuracy_first_lexicographic(acc,macro_f1)",
    }:
        return THRESHOLD_OBJECTIVE_ACC_FIRST
    if raw in {
        THRESHOLD_OBJECTIVE_F1_WITH_ACC_FLOOR,
        "macro_f1_first",
        "macro_f1_priority",
    }:
        return THRESHOLD_OBJECTIVE_F1_WITH_ACC_FLOOR
    raise ValueError(
        "threshold_search_objective 仅支持 "
        f"{THRESHOLD_OBJECTIVE_ACC_FIRST}/{THRESHOLD_OBJECTIVE_F1_WITH_ACC_FLOOR}，当前={objective!r}"
    )


def _objective_to_payload_text(objective: str, min_accuracy: float) -> str:
    normalized = _normalize_threshold_search_objective(objective)
    if normalized == THRESHOLD_OBJECTIVE_ACC_FIRST:
        return "accuracy_first_lexicographic(acc,macro_f1)"
    return f"macro_f1_with_accuracy_floor(acc>={float(min_accuracy):.3f})"


def _save_confusion_matrix_heatmap(
    cm: np.ndarray,
    output_path: Path,
    title: str,
    class_names: List[str],
) -> Optional[Path]:
    """
    将混淆矩阵保存为热力图 PNG 文件。

    设计说明：
    1. 评估脚本的核心职责是“测试后可解释输出”，因此这里直接在评估阶段落盘图像，
       避免后处理脚本遗漏，保证每次评估都有一致产物。
    2. 使用 Agg 后端，确保在无图形界面的服务器/WSL 环境也能稳定保存图片。
    3. 对空矩阵直接跳过并返回 None，避免无意义输出和潜在异常。
    """

    # 空矩阵不绘图：这通常意味着测试集为空或评估流程提前返回。
    if cm.size == 0:
        return None

    try:
        # 在函数内延迟导入 matplotlib，避免将其作为训练/评估的强依赖。
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 依赖缺失时仅做容错打印
        print(f"[评估] 跳过混淆矩阵热力图导出：matplotlib 不可用 ({exc})")
        return None

    if len(class_names) != cm.shape[0]:
        raise ValueError(
            f"class_names 长度({len(class_names)})必须等于混淆矩阵行数({cm.shape[0]})"
        )
    for idx, class_name in enumerate(class_names):
        # 约束：类别标签必须是语义名称，不能是“0/1/2...”这类纯数字占位符。
        # 这样做是为了保证热力图可解释性，避免再次出现“只有数字索引”的问题。
        if str(class_name).strip().isdigit():
            raise ValueError(
                f"class_names[{idx}]={class_name!r} 不能为纯数字，请提供类别名称"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=14)

    x_ticks = np.arange(cm.shape[1])
    y_ticks = np.arange(cm.shape[0])
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=30)

    # 数字标注可直接反映错分模式；阈值分色用于保持深浅背景下的可读性。
    threshold = float(cm.max()) * 0.5 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            text_color = "white" if value > threshold else "black"
            ax.text(j, i, f"{value}", ha="center", va="center", color=text_color, fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _build_class_names_from_config() -> List[str]:
    """
    从全局配置中构建“类别索引 -> 类别名称”列表。

    设计原因：
    1. 将映射逻辑集中到单一函数，避免在评估流程里散落重复代码。
    2. 对缺失映射做显式失败，防止热力图悄悄退化为数字标签。
    """

    class_names: List[str] = []
    for class_id in range(CFG.num_classes):
        if class_id not in CFG.class_id_to_name:
            raise KeyError(
                f"class_id_to_name 缺少类别 {class_id} 的名称映射，无法绘制类别名热力图"
            )
        class_name = str(CFG.class_id_to_name[class_id]).strip()
        if not class_name:
            raise ValueError(f"class_id_to_name[{class_id}] 不能为空字符串")
        class_names.append(class_name)
    return class_names


def _compute_per_class_metrics(
    cm: np.ndarray,
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """
    从混淆矩阵计算每类 precision / recall / f1。
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm 必须是方阵，当前={cm.shape}")
    if len(class_names) != cm.shape[0]:
        raise ValueError(
            f"class_names 长度({len(class_names)})必须等于类别数({cm.shape[0]})"
        )

    cm_int = cm.astype(np.int64, copy=False)
    support = cm_int.sum(axis=1)
    predicted = cm_int.sum(axis=0)
    per_class: List[Dict[str, Any]] = []
    for class_id in range(cm_int.shape[0]):
        tp = int(cm_int[class_id, class_id])
        sup = int(support[class_id])
        pred = int(predicted[class_id])
        precision = float(tp / pred) if pred > 0 else 0.0
        recall = float(tp / sup) if sup > 0 else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )
        per_class.append(
            {
                "class_id": class_id,
                "class_name": str(class_names[class_id]),
                "support": sup,
                "predicted": pred,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return per_class


def _compute_top_confusion_pairs(
    cm: np.ndarray,
    class_names: List[str],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    统计最严重的非对角误分类对。
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm 必须是方阵，当前={cm.shape}")
    if len(class_names) != cm.shape[0]:
        raise ValueError(
            f"class_names 长度({len(class_names)})必须等于类别数({cm.shape[0]})"
        )

    cm_int = cm.astype(np.int64, copy=False)
    row_sum = cm_int.sum(axis=1)
    pairs: List[Dict[str, Any]] = []
    for true_class in range(cm_int.shape[0]):
        for pred_class in range(cm_int.shape[1]):
            if true_class == pred_class:
                continue
            count = int(cm_int[true_class, pred_class])
            if count <= 0:
                continue
            denom = int(row_sum[true_class])
            row_rate = float(count / denom) if denom > 0 else 0.0
            pairs.append(
                {
                    "true_class_id": int(true_class),
                    "true_class_name": str(class_names[true_class]),
                    "pred_class_id": int(pred_class),
                    "pred_class_name": str(class_names[pred_class]),
                    "count": count,
                    "row_rate": row_rate,
                }
            )
    pairs.sort(key=lambda x: (int(x["count"]), float(x["row_rate"])), reverse=True)
    return pairs[: int(max(top_k, 0))]


def _build_eval_metrics_payload(result: EvalResult) -> Dict[str, Any]:
    """
    提取评估结果中便于摘要/榜单复用的核心字段。
    """
    return {
        "model_name": str(result.model_name),
        "domain": str(result.domain),
        "split": str(result.split),
        "accuracy": float(result.accuracy),
        "macro_f1": float(result.macro_f1),
        "macro_recall": float(result.macro_recall),
        "weighted_f1": float(result.weighted_f1),
        "gmean": float(result.gmean),
        "threshold_mode": result.threshold_mode,
        "threshold_applied": result.threshold_applied,
        "global_threshold": result.global_threshold,
        "macro_f1_before_threshold": result.macro_f1_before_threshold,
        "per_class_thresholds": result.per_class_thresholds,
        "threshold_selection_reason": result.threshold_selection_reason,
        "probability_temperature": result.probability_temperature,
        "nofault_logit_bias": result.nofault_logit_bias,
    }


def _write_target_test_leaderboard(
    eval_dir: Path,
    *,
    raw_result: EvalResult,
    thresholded_result: EvalResult,
    objective_text: str = "accuracy_first_lexicographic(acc,macro_f1)",
    selection_stability_score: Optional[float] = None,
    threshold_constraint_flags: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    生成统一榜单文件（同一次 run 的 raw/thresholded 双记录）。
    """
    entries: List[Dict[str, Any]] = [
        {
            "entry_type": "raw",
            "metrics": _build_eval_metrics_payload(raw_result),
            "per_class_metrics": raw_result.per_class_metrics,
            "top_confusion_pairs": raw_result.top_confusion_pairs,
        },
        {
            "entry_type": "thresholded",
            "metrics": _build_eval_metrics_payload(thresholded_result),
            "per_class_metrics": thresholded_result.per_class_metrics,
            "top_confusion_pairs": thresholded_result.top_confusion_pairs,
            "selection_stability_score": (
                None
                if selection_stability_score is None
                else float(selection_stability_score)
            ),
            "threshold_constraint_flags": (
                None
                if threshold_constraint_flags is None
                else dict(threshold_constraint_flags)
            ),
        },
    ]
    ranked = sorted(
        entries,
        key=lambda item: (
            float(item["metrics"]["accuracy"]),
            float(item["metrics"]["macro_f1"]),
        ),
        reverse=True,
    )
    for rank, entry in enumerate(ranked, start=1):
        entry["rank_accuracy_first"] = int(rank)

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": str(objective_text),
        "entries": ranked,
    }
    leaderboard_path = eval_dir / "leaderboard_target_test.json"
    write_json(leaderboard_path, payload)
    return leaderboard_path


def _load_domain_artifacts(
    run_dir: Path,
    domain: str,
    split: str = "test",
):
    """加载指定域的分割数据（test/val）。"""
    split_name = str(split).strip().lower()
    if split_name not in {"test", "val"}:
        raise ValueError(f"split 仅支持 test/val，当前={split}")

    test_path = run_dir / f"split_{domain}_{split_name}.json"
    stats_path = run_dir / "source_stats.json"

    if not test_path.exists():
        raise FileNotFoundError(f"缺少 split_{domain}_{split_name}.json")
    if not stats_path.exists():
        raise FileNotFoundError("缺少 source_stats.json")

    test_records = read_json(test_path)
    stats = read_json(stats_path)

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    return test_records, mean, std


def _apply_global_confidence_threshold(
    y_prob: np.ndarray,
    threshold: float,
    fallback_class_id: int,
) -> np.ndarray:
    """
    对多分类预测应用全局置信度阈值。

    规则:
    - 若 `max(prob) < threshold`，则回退到 `fallback_class_id`；
    - 否则保持 argmax 预测。
    """
    if y_prob.ndim != 2:
        raise ValueError(f"y_prob 必须是二维数组 [N,C]，当前={y_prob.shape}")
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"threshold 必须在 [0,1]，当前={threshold}")

    pred = np.argmax(y_prob, axis=1).astype(np.int64)
    confidence = np.max(y_prob, axis=1)
    pred[confidence < float(threshold)] = int(fallback_class_id)
    return pred


def _apply_per_class_thresholds(
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    对多分类预测应用按类别阈值校准。

    规则:
    - 对每个类别 c 使用 score_c = prob_c - threshold_c；
    - 预测 score 最大的类别。
    """
    if y_prob.ndim != 2:
        raise ValueError(f"y_prob 必须是二维数组 [N,C]，当前={y_prob.shape}")
    thresholds_arr = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if thresholds_arr.size != y_prob.shape[1]:
        raise ValueError(
            f"thresholds 长度({thresholds_arr.size})必须等于类别数({y_prob.shape[1]})"
        )
    adjusted = y_prob - thresholds_arr[np.newaxis, :]
    return np.argmax(adjusted, axis=1).astype(np.int64)


def _compute_candidate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class10_id: int,
    class10_prior: float,
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_recall = float(
        recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    )
    accuracy = float(np.mean((y_true_arr == y_pred_arr).astype(np.float32)))
    gmean = float(compute_gmean(y_true_arr, y_pred_arr))
    pred_rate_class10 = float(np.mean((y_pred_arr == int(class10_id)).astype(np.float32)))
    class10_overpredict = float(max(0.0, pred_rate_class10 - float(class10_prior)))
    return {
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "accuracy": accuracy,
        "gmean": gmean,
        "pred_rate_class10": pred_rate_class10,
        "class10_overpredict": class10_overpredict,
    }


def _is_candidate_better(
    challenger: Dict[str, Any],
    incumbent: Dict[str, Any],
    *,
    class10_constraint: str,
    threshold_search_objective: str = THRESHOLD_OBJECTIVE_ACC_FIRST,
    threshold_search_min_accuracy: float = 0.0,
    accuracy_tie_eps: float = 0.0,
    eps: float = 1e-12,
) -> bool:
    del class10_constraint
    objective = _normalize_threshold_search_objective(threshold_search_objective)
    tie_eps = max(float(accuracy_tie_eps), float(eps))
    challenger_acc = float(challenger["accuracy_after_threshold"])
    incumbent_acc = float(incumbent["accuracy_after_threshold"])
    challenger_f1 = float(challenger["macro_f1_after_threshold"])
    incumbent_f1 = float(incumbent["macro_f1_after_threshold"])

    if objective == THRESHOLD_OBJECTIVE_F1_WITH_ACC_FLOOR:
        # 宏 F1 优先，但仅在满足准确率下限时参与竞争。
        min_acc = float(threshold_search_min_accuracy)
        challenger_meets_floor = challenger_acc + eps >= min_acc
        incumbent_meets_floor = incumbent_acc + eps >= min_acc
        if challenger_meets_floor != incumbent_meets_floor:
            return bool(challenger_meets_floor)
        if challenger_f1 > incumbent_f1 + eps:
            return True
        if abs(challenger_f1 - incumbent_f1) > eps:
            return False
        if challenger_acc > incumbent_acc + tie_eps:
            return True
        if abs(challenger_acc - incumbent_acc) > tie_eps:
            return False
    else:
        # Accuracy-first lexicographic objective:
        # 1) maximize accuracy
        # 2) within tie-eps, maximize macro-F1
        if challenger_acc > incumbent_acc + tie_eps:
            return True
        if abs(challenger_acc - incumbent_acc) > tie_eps:
            return False
        if challenger_f1 > incumbent_f1 + eps:
            return True
        if abs(challenger_f1 - incumbent_f1) > eps:
            return False

    challenger_recall = float(challenger.get("macro_recall_after_threshold", 0.0))
    incumbent_recall = float(incumbent.get("macro_recall_after_threshold", 0.0))
    if challenger_recall > incumbent_recall + eps:
        return True
    if abs(challenger_recall - incumbent_recall) > eps:
        return False

    challenger_gmean = float(challenger.get("gmean_after_threshold", 0.0))
    incumbent_gmean = float(incumbent.get("gmean_after_threshold", 0.0))
    if challenger_gmean > incumbent_gmean + eps:
        return True
    if abs(challenger_gmean - incumbent_gmean) > eps:
        return False

    if float(challenger["class10_overpredict_after"]) < float(incumbent["class10_overpredict_after"]) - eps:
        return True
    return False


def _build_threshold_constraint_flags(
    *,
    baseline_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
    min_macro_recall_delta: float,
    min_gmean_delta: float,
) -> Dict[str, Any]:
    baseline_macro_recall = float(baseline_metrics.get("macro_recall", 0.0))
    baseline_gmean = float(baseline_metrics.get("gmean", 0.0))
    candidate_macro_recall = float(candidate_metrics.get("macro_recall", 0.0))
    candidate_gmean = float(candidate_metrics.get("gmean", 0.0))

    macro_recall_delta = candidate_macro_recall - baseline_macro_recall
    gmean_delta = candidate_gmean - baseline_gmean
    macro_recall_constraint_pass = macro_recall_delta >= float(min_macro_recall_delta) - 1e-12
    gmean_constraint_pass = gmean_delta >= float(min_gmean_delta) - 1e-12
    return {
        "macro_recall_delta": float(macro_recall_delta),
        "gmean_delta": float(gmean_delta),
        "min_macro_recall_delta": float(min_macro_recall_delta),
        "min_gmean_delta": float(min_gmean_delta),
        "macro_recall_constraint_pass": bool(macro_recall_constraint_pass),
        "gmean_constraint_pass": bool(gmean_constraint_pass),
    }


def _is_candidate_within_constraints(constraint_flags: Dict[str, Any]) -> bool:
    return bool(
        constraint_flags.get("macro_recall_constraint_pass", False)
        and constraint_flags.get("gmean_constraint_pass", False)
    )


def search_best_global_confidence_threshold_candidate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fallback_class_id: int,
    *,
    class10_id: int,
    class10_prior: float,
    max_accuracy_drop: float = 0.0,
    class10_constraint: str = "soft",
    class10_overpredict_limit: float = 0.0,
    min_macro_recall_delta: float = -0.01,
    min_gmean_delta: float = -0.02,
    threshold_search_objective: str = THRESHOLD_OBJECTIVE_ACC_FIRST,
    threshold_search_min_accuracy: float = 0.0,
    accuracy_tie_eps: float = 0.0,
    num_thresholds: int = 101,
) -> Dict[str, Any]:
    """
    搜索满足约束的全局阈值候选。
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    if y_true_arr.size == 0:
        return {
            "mode": "global",
            "is_valid": False,
            "selection_reason": "target_val_empty",
        }

    raw_pred = np.argmax(y_prob_arr, axis=1).astype(np.int64)
    base = _compute_candidate_metrics(
        y_true_arr,
        raw_pred,
        class10_id=class10_id,
        class10_prior=class10_prior,
    )
    normalized_objective = _normalize_threshold_search_objective(
        threshold_search_objective
    )
    min_accuracy_floor = float(threshold_search_min_accuracy)

    best: Optional[Dict[str, Any]] = None
    min_allowed_acc = float(base["accuracy"] - float(max_accuracy_drop))
    for threshold in np.linspace(0.0, 1.0, int(max(num_thresholds, 2))):
        pred = _apply_global_confidence_threshold(
            y_prob=y_prob_arr,
            threshold=float(threshold),
            fallback_class_id=int(fallback_class_id),
        )
        cand_metrics = _compute_candidate_metrics(
            y_true_arr,
            pred,
            class10_id=class10_id,
            class10_prior=class10_prior,
        )
        constraint_flags = _build_threshold_constraint_flags(
            baseline_metrics=base,
            candidate_metrics=cand_metrics,
            min_macro_recall_delta=min_macro_recall_delta,
            min_gmean_delta=min_gmean_delta,
        )
        if float(cand_metrics["accuracy"]) + 1e-12 < min_allowed_acc:
            continue
        if (
            normalized_objective == THRESHOLD_OBJECTIVE_F1_WITH_ACC_FLOOR
            and float(cand_metrics["accuracy"]) + 1e-12 < min_accuracy_floor
        ):
            continue
        if (
            class10_constraint == "hard"
            and float(cand_metrics["pred_rate_class10"])
            > float(class10_prior) + float(class10_overpredict_limit) + 1e-12
        ):
            continue
        if not _is_candidate_within_constraints(constraint_flags):
            continue

        candidate = {
            "mode": "global",
            "best_threshold": float(threshold),
            "macro_f1_before_threshold": float(base["macro_f1"]),
            "macro_recall_before_threshold": float(base["macro_recall"]),
            "gmean_before_threshold": float(base["gmean"]),
            "accuracy_before_threshold": float(base["accuracy"]),
            "macro_f1_after_threshold": float(cand_metrics["macro_f1"]),
            "macro_recall_after_threshold": float(cand_metrics["macro_recall"]),
            "gmean_after_threshold": float(cand_metrics["gmean"]),
            "accuracy_after_threshold": float(cand_metrics["accuracy"]),
            "pred_rate_class10_before": float(base["pred_rate_class10"]),
            "pred_rate_class10_after": float(cand_metrics["pred_rate_class10"]),
            "class10_prior": float(class10_prior),
            "class10_overpredict_after": float(cand_metrics["class10_overpredict"]),
            "class10_constraint": str(class10_constraint),
            "class10_overpredict_limit": float(class10_overpredict_limit),
            "max_accuracy_drop": float(max_accuracy_drop),
            "threshold_constraint_flags": constraint_flags,
            "is_valid": True,
            "selection_reason": "ok",
        }
        if best is None or _is_candidate_better(
            challenger=candidate,
            incumbent=best,
            class10_constraint=str(class10_constraint),
            threshold_search_objective=normalized_objective,
            threshold_search_min_accuracy=min_accuracy_floor,
            accuracy_tie_eps=float(accuracy_tie_eps),
        ):
            best = candidate

    if best is None:
        return {
            "mode": "global",
            "is_valid": False,
            "selection_reason": "no_feasible_candidate",
            "macro_f1_before_threshold": float(base["macro_f1"]),
            "macro_recall_before_threshold": float(base["macro_recall"]),
            "gmean_before_threshold": float(base["gmean"]),
            "accuracy_before_threshold": float(base["accuracy"]),
        }
    return best


def search_best_per_class_confidence_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    class10_id: int,
    class10_prior: float,
    max_accuracy_drop: float = 0.0,
    class10_constraint: str = "soft",
    class10_overpredict_limit: float = 0.0,
    min_macro_recall_delta: float = -0.01,
    min_gmean_delta: float = -0.02,
    threshold_search_objective: str = THRESHOLD_OBJECTIVE_ACC_FIRST,
    threshold_search_min_accuracy: float = 0.0,
    accuracy_tie_eps: float = 0.0,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    base_grid: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55),
    gamma_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    refine_rounds: int = 2,
    refine_radius: float = 0.15,
    refine_step: float = 0.05,
) -> Dict[str, Any]:
    """
    基于验证集类别先验搜索 per-class 阈值，满足准确率约束并最大化 macro-F1。
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    if y_true_arr.size == 0:
        return {
            "mode": "per_class",
            "is_valid": False,
            "selection_reason": "target_val_empty",
        }
    if y_prob_arr.ndim != 2:
        raise ValueError(f"y_prob 必须是二维数组 [N,C]，当前={y_prob_arr.shape}")

    num_classes = int(y_prob_arr.shape[1])
    counts = np.bincount(y_true_arr, minlength=num_classes).astype(np.float64)
    priors = counts / max(float(y_true_arr.size), 1.0)
    uniform_prior = 1.0 / max(float(num_classes), 1.0)

    raw_pred = np.argmax(y_prob_arr, axis=1).astype(np.int64)
    base = _compute_candidate_metrics(
        y_true_arr,
        raw_pred,
        class10_id=class10_id,
        class10_prior=class10_prior,
    )
    normalized_objective = _normalize_threshold_search_objective(
        threshold_search_objective
    )
    min_accuracy_floor = float(threshold_search_min_accuracy)
    min_allowed_acc = float(base["accuracy"] - float(max_accuracy_drop))

    best: Optional[Dict[str, Any]] = None

    def _try_candidate(thresholds_arr: np.ndarray) -> None:
        nonlocal best
        pred = _apply_per_class_thresholds(y_prob_arr, thresholds_arr)
        metrics = _compute_candidate_metrics(
            y_true_arr,
            pred,
            class10_id=class10_id,
            class10_prior=class10_prior,
        )
        constraint_flags = _build_threshold_constraint_flags(
            baseline_metrics=base,
            candidate_metrics=metrics,
            min_macro_recall_delta=min_macro_recall_delta,
            min_gmean_delta=min_gmean_delta,
        )
        if float(metrics["accuracy"]) + 1e-12 < min_allowed_acc:
            return
        if (
            normalized_objective == THRESHOLD_OBJECTIVE_F1_WITH_ACC_FLOOR
            and float(metrics["accuracy"]) + 1e-12 < min_accuracy_floor
        ):
            return
        if (
            class10_constraint == "hard"
            and float(metrics["pred_rate_class10"])
            > float(class10_prior) + float(class10_overpredict_limit) + 1e-12
        ):
            return
        if not _is_candidate_within_constraints(constraint_flags):
            return
        candidate = {
            "mode": "per_class",
            "per_class_thresholds": thresholds_arr.astype(np.float32).tolist(),
            "macro_f1_before_threshold": float(base["macro_f1"]),
            "macro_recall_before_threshold": float(base["macro_recall"]),
            "gmean_before_threshold": float(base["gmean"]),
            "accuracy_before_threshold": float(base["accuracy"]),
            "macro_f1_after_threshold": float(metrics["macro_f1"]),
            "macro_recall_after_threshold": float(metrics["macro_recall"]),
            "gmean_after_threshold": float(metrics["gmean"]),
            "accuracy_after_threshold": float(metrics["accuracy"]),
            "pred_rate_class10_before": float(base["pred_rate_class10"]),
            "pred_rate_class10_after": float(metrics["pred_rate_class10"]),
            "class10_prior": float(class10_prior),
            "class10_overpredict_after": float(metrics["class10_overpredict"]),
            "class10_constraint": str(class10_constraint),
            "class10_overpredict_limit": float(class10_overpredict_limit),
            "max_accuracy_drop": float(max_accuracy_drop),
            "threshold_constraint_flags": constraint_flags,
            "is_valid": True,
            "selection_reason": "ok",
        }
        if best is None or _is_candidate_better(
            challenger=candidate,
            incumbent=best,
            class10_constraint=str(class10_constraint),
            threshold_search_objective=normalized_objective,
            threshold_search_min_accuracy=min_accuracy_floor,
            accuracy_tie_eps=float(accuracy_tie_eps),
        ):
            best = candidate

    for base_threshold in base_grid:
        for gamma in gamma_grid:
            scaled = float(base_threshold) * np.power(
                np.clip(priors / max(uniform_prior, 1e-12), 1e-12, None),
                float(gamma),
            )
            thresholds = np.clip(scaled, float(threshold_min), float(threshold_max))
            _try_candidate(thresholds.astype(np.float32))

    if best is None:
        return {
            "mode": "per_class",
            "is_valid": False,
            "selection_reason": "no_feasible_candidate",
            "macro_f1_before_threshold": float(base["macro_f1"]),
            "macro_recall_before_threshold": float(base["macro_recall"]),
            "gmean_before_threshold": float(base["gmean"]),
            "accuracy_before_threshold": float(base["accuracy"]),
        }

    current = np.asarray(best["per_class_thresholds"], dtype=np.float32)
    radius = float(refine_radius)
    for _ in range(int(max(refine_rounds, 0))):
        for class_id in range(num_classes):
            lo = max(float(threshold_min), float(current[class_id] - radius))
            hi = min(float(threshold_max), float(current[class_id] + radius))
            values = np.arange(lo, hi + 1e-9, float(refine_step), dtype=np.float32)
            for value in values.tolist():
                candidate_thresholds = current.copy()
                candidate_thresholds[class_id] = float(value)
                _try_candidate(candidate_thresholds)
        if best is not None:
            current = np.asarray(best["per_class_thresholds"], dtype=np.float32)
        radius = max(float(refine_step), float(radius * 0.5))

    return best


def _softmax_with_temperature(
    raw_logits: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    logits_arr = np.asarray(raw_logits, dtype=np.float32)
    if logits_arr.ndim != 2:
        raise ValueError(f"raw_logits 必须是二维数组 [N,C]，当前={logits_arr.shape}")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature 必须 > 0，当前={temperature}")
    scaled = logits_arr / float(temperature)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled).astype(np.float32)
    denom = np.sum(exp_scaled, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return (exp_scaled / denom).astype(np.float32)


def _apply_nofault_logit_bias_to_logits(
    raw_logits: np.ndarray,
    *,
    class10_id: int,
    nofault_logit_bias: float,
    temperature: float = 1.0,
) -> np.ndarray:
    logits_arr = np.asarray(raw_logits, dtype=np.float32)
    if logits_arr.ndim != 2:
        raise ValueError(f"raw_logits 必须是二维数组 [N,C]，当前={logits_arr.shape}")
    class10_idx = int(class10_id)
    if class10_idx < 0 or class10_idx >= logits_arr.shape[1]:
        raise ValueError(
            f"class10_id 必须在 [0,{logits_arr.shape[1] - 1}]，当前={class10_id}"
        )
    adjusted_logits = logits_arr.copy()
    adjusted_logits[:, class10_idx] -= float(nofault_logit_bias)
    return _softmax_with_temperature(adjusted_logits, temperature=float(temperature))


def _apply_nofault_logit_bias_to_probabilities(
    y_prob: np.ndarray,
    *,
    class10_id: int,
    nofault_logit_bias: float,
    temperature: float = 1.0,
) -> np.ndarray:
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    if y_prob_arr.ndim != 2:
        raise ValueError(f"y_prob 必须是二维数组 [N,C]，当前={y_prob_arr.shape}")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature 必须 > 0，当前={temperature}")
    class10_idx = int(class10_id)
    if class10_idx < 0 or class10_idx >= y_prob_arr.shape[1]:
        raise ValueError(
            f"class10_id 必须在 [0,{y_prob_arr.shape[1] - 1}]，当前={class10_id}"
        )
    adjusted = y_prob_arr.copy()
    multiplier = float(np.exp(-float(nofault_logit_bias) / float(temperature)))
    adjusted[:, class10_idx] *= multiplier
    denom = np.sum(adjusted, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return (adjusted / denom).astype(np.float32)


def search_best_nofault_logit_bias(
    y_true: np.ndarray,
    raw_logits: np.ndarray,
    *,
    class10_id: int,
    class10_prior: float,
    temperature: float = 1.0,
    max_accuracy_drop: float = 0.0,
    class10_constraint: str = "soft",
    class10_overpredict_limit: float = 0.0,
    min_macro_recall_delta: float = -0.01,
    min_gmean_delta: float = -0.02,
    threshold_search_objective: str = THRESHOLD_OBJECTIVE_ACC_FIRST,
    threshold_search_min_accuracy: float = 0.0,
    accuracy_tie_eps: float = 0.0,
    bias_min: float = 0.0,
    bias_max: float = 5.0,
    bias_step: float = 0.1,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    logits_arr = np.asarray(raw_logits, dtype=np.float32)
    if y_true_arr.size == 0:
        return {
            "mode": "nofault_bias",
            "is_valid": False,
            "selection_reason": "target_val_empty",
        }
    if logits_arr.ndim != 2:
        raise ValueError(f"raw_logits 必须是二维数组 [N,C]，当前={logits_arr.shape}")
    if logits_arr.shape[0] != y_true_arr.shape[0]:
        raise ValueError(
            "raw_logits 样本数必须与 y_true 一致，"
            f"当前 raw_logits={logits_arr.shape[0]}, y_true={y_true_arr.shape[0]}"
        )
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature 必须 > 0，当前={temperature}")
    if float(bias_step) <= 0.0:
        raise ValueError(f"bias_step 必须 > 0，当前={bias_step}")
    if float(bias_max) < float(bias_min):
        raise ValueError(f"bias_max 必须 >= bias_min，当前={bias_max} < {bias_min}")

    base_prob = _softmax_with_temperature(logits_arr, temperature=float(temperature))
    base_pred = np.argmax(base_prob, axis=1).astype(np.int64)
    base = _compute_candidate_metrics(
        y_true_arr,
        base_pred,
        class10_id=class10_id,
        class10_prior=class10_prior,
    )
    normalized_objective = _normalize_threshold_search_objective(
        threshold_search_objective
    )
    min_accuracy_floor = float(threshold_search_min_accuracy)
    min_allowed_acc = float(base["accuracy"] - float(max_accuracy_drop))

    bias_values = np.arange(
        float(bias_min),
        float(bias_max) + float(bias_step) * 0.5,
        float(bias_step),
        dtype=np.float32,
    )

    best: Optional[Dict[str, Any]] = None
    for nofault_bias in bias_values.tolist():
        adjusted_prob = _apply_nofault_logit_bias_to_logits(
            logits_arr,
            class10_id=int(class10_id),
            nofault_logit_bias=float(nofault_bias),
            temperature=float(temperature),
        )
        pred = np.argmax(adjusted_prob, axis=1).astype(np.int64)
        cand_metrics = _compute_candidate_metrics(
            y_true_arr,
            pred,
            class10_id=class10_id,
            class10_prior=class10_prior,
        )
        constraint_flags = _build_threshold_constraint_flags(
            baseline_metrics=base,
            candidate_metrics=cand_metrics,
            min_macro_recall_delta=min_macro_recall_delta,
            min_gmean_delta=min_gmean_delta,
        )
        if float(cand_metrics["accuracy"]) + 1e-12 < min_allowed_acc:
            continue
        if (
            normalized_objective == THRESHOLD_OBJECTIVE_F1_WITH_ACC_FLOOR
            and float(cand_metrics["accuracy"]) + 1e-12 < min_accuracy_floor
        ):
            continue
        if (
            class10_constraint == "hard"
            and float(cand_metrics["pred_rate_class10"])
            > float(class10_prior) + float(class10_overpredict_limit) + 1e-12
        ):
            continue
        if not _is_candidate_within_constraints(constraint_flags):
            continue
        candidate = {
            "mode": "nofault_bias",
            "nofault_logit_bias": float(nofault_bias),
            "macro_f1_before_threshold": float(base["macro_f1"]),
            "macro_recall_before_threshold": float(base["macro_recall"]),
            "gmean_before_threshold": float(base["gmean"]),
            "accuracy_before_threshold": float(base["accuracy"]),
            "macro_f1_after_threshold": float(cand_metrics["macro_f1"]),
            "macro_recall_after_threshold": float(cand_metrics["macro_recall"]),
            "gmean_after_threshold": float(cand_metrics["gmean"]),
            "accuracy_after_threshold": float(cand_metrics["accuracy"]),
            "pred_rate_class10_before": float(base["pred_rate_class10"]),
            "pred_rate_class10_after": float(cand_metrics["pred_rate_class10"]),
            "class10_prior": float(class10_prior),
            "class10_overpredict_after": float(cand_metrics["class10_overpredict"]),
            "class10_constraint": str(class10_constraint),
            "class10_overpredict_limit": float(class10_overpredict_limit),
            "max_accuracy_drop": float(max_accuracy_drop),
            "temperature": float(temperature),
            "threshold_constraint_flags": constraint_flags,
            "is_valid": True,
            "selection_reason": "ok",
        }
        if best is None or _is_candidate_better(
            challenger=candidate,
            incumbent=best,
            class10_constraint=str(class10_constraint),
            threshold_search_objective=normalized_objective,
            threshold_search_min_accuracy=min_accuracy_floor,
            accuracy_tie_eps=float(accuracy_tie_eps),
        ):
            best = candidate

    if best is None:
        return {
            "mode": "nofault_bias",
            "is_valid": False,
            "selection_reason": "no_feasible_candidate",
            "macro_f1_before_threshold": float(base["macro_f1"]),
            "macro_recall_before_threshold": float(base["macro_recall"]),
            "gmean_before_threshold": float(base["gmean"]),
            "accuracy_before_threshold": float(base["accuracy"]),
        }
    return best


def _select_threshold_candidate(
    *,
    baseline_macro_f1: float,
    baseline_macro_recall: float = 0.0,
    baseline_gmean: float = 0.0,
    class10_constraint: str,
    require_non_decreasing_macro_f1: bool,
    threshold_search_objective: str = THRESHOLD_OBJECTIVE_ACC_FIRST,
    threshold_search_min_accuracy: float = 0.0,
    min_macro_recall_delta: float = -0.01,
    min_gmean_delta: float = -0.02,
    accuracy_tie_eps: float = 0.0,
    global_candidate: Optional[Dict[str, Any]],
    per_class_candidate: Optional[Dict[str, Any]],
    nofault_bias_candidate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    统一比较 global/per-class 候选，输出最终选择。
    """
    normalized_objective = _normalize_threshold_search_objective(
        threshold_search_objective
    )
    min_accuracy_floor = float(threshold_search_min_accuracy)
    available: List[Dict[str, Any]] = []
    for candidate in (global_candidate, per_class_candidate, nofault_bias_candidate):
        if not isinstance(candidate, dict):
            continue
        if not bool(candidate.get("is_valid", False)):
            continue
        if (
            normalized_objective == THRESHOLD_OBJECTIVE_F1_WITH_ACC_FLOOR
            and float(candidate.get("accuracy_after_threshold", 0.0)) + 1e-12
            < min_accuracy_floor
        ):
            continue
        constraint_flags = candidate.get("threshold_constraint_flags")
        if not isinstance(constraint_flags, dict):
            constraint_flags = _build_threshold_constraint_flags(
                baseline_metrics={
                    "macro_recall": float(baseline_macro_recall),
                    "gmean": float(baseline_gmean),
                },
                candidate_metrics={
                    "macro_recall": float(candidate.get("macro_recall_after_threshold", 0.0)),
                    "gmean": float(candidate.get("gmean_after_threshold", 0.0)),
                },
                min_macro_recall_delta=min_macro_recall_delta,
                min_gmean_delta=min_gmean_delta,
            )
            candidate["threshold_constraint_flags"] = constraint_flags
        if not _is_candidate_within_constraints(constraint_flags):
            continue
        if (
            bool(require_non_decreasing_macro_f1)
            and float(candidate.get("macro_f1_after_threshold", 0.0))
            < float(baseline_macro_f1) - 1e-12
        ):
            continue
        available.append(candidate)

    if len(available) == 0:
        return {
            "selected_mode": "none",
            "selected_candidate": None,
            "threshold_constraint_flags": None,
            "selection_reason": (
                "no_candidate_meets_constraints_and_macro_f1_non_decrease"
                if bool(require_non_decreasing_macro_f1)
                else "no_candidate_meets_constraints"
            ),
        }

    best = available[0]
    for candidate in available[1:]:
        if _is_candidate_better(
            challenger=candidate,
            incumbent=best,
            class10_constraint=str(class10_constraint),
            threshold_search_objective=normalized_objective,
            threshold_search_min_accuracy=min_accuracy_floor,
            accuracy_tie_eps=float(accuracy_tie_eps),
        ):
            best = candidate
    return {
        "selected_mode": str(best["mode"]),
        "selected_candidate": best,
        "threshold_constraint_flags": best.get("threshold_constraint_flags"),
        "selection_reason": "ok",
    }


def _bootstrap_selection_stability(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class10_id: int,
    fallback_class_id: int,
    class10_constraint: str,
    require_non_decreasing_macro_f1: bool,
    threshold_search_objective: str = THRESHOLD_OBJECTIVE_ACC_FIRST,
    threshold_search_min_accuracy: float = 0.0,
    min_macro_recall_delta: float,
    min_gmean_delta: float,
    accuracy_tie_eps: float,
    baseline_metrics: Dict[str, float],
    global_candidate: Optional[Dict[str, Any]],
    per_class_candidate: Optional[Dict[str, Any]],
    selected_mode: str,
    rounds: int,
    nofault_bias_candidate: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    probability_temperature: float = 1.0,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    rounds_int = int(max(rounds, 0))
    mode_counts: Dict[str, int] = {
        "global": 0,
        "per_class": 0,
        "nofault_bias": 0,
        "none": 0,
    }
    if rounds_int == 0 or y_true_arr.size == 0:
        return {
            "rounds": rounds_int,
            "seed": int(seed),
            "mode_counts": mode_counts,
            "selected_mode": str(selected_mode),
            "selection_stability_score": 0.0,
        }

    raw_pred = np.argmax(y_prob_arr, axis=1).astype(np.int64)
    cached_pred: Dict[str, np.ndarray] = {"raw": raw_pred}
    if isinstance(global_candidate, dict) and bool(global_candidate.get("is_valid", False)):
        threshold = float(global_candidate.get("best_threshold", 0.0))
        cached_pred["global"] = _apply_global_confidence_threshold(
            y_prob=y_prob_arr,
            threshold=threshold,
            fallback_class_id=int(fallback_class_id),
        )
    if isinstance(per_class_candidate, dict) and bool(per_class_candidate.get("is_valid", False)):
        per_class_thresholds = np.asarray(
            per_class_candidate.get("per_class_thresholds", []),
            dtype=np.float32,
        )
        if per_class_thresholds.size == y_prob_arr.shape[1]:
            cached_pred["per_class"] = _apply_per_class_thresholds(
                y_prob=y_prob_arr,
                thresholds=per_class_thresholds,
            )
    if isinstance(nofault_bias_candidate, dict) and bool(nofault_bias_candidate.get("is_valid", False)):
        nofault_logit_bias = float(nofault_bias_candidate.get("nofault_logit_bias", 0.0))
        adjusted_prob = _apply_nofault_logit_bias_to_probabilities(
            y_prob=y_prob_arr,
            class10_id=int(class10_id),
            nofault_logit_bias=nofault_logit_bias,
            temperature=float(probability_temperature),
        )
        cached_pred["nofault_bias"] = np.argmax(adjusted_prob, axis=1).astype(np.int64)

    rng = np.random.default_rng(seed=int(seed))
    n = int(y_true_arr.size)
    for _ in range(rounds_int):
        sample_idx = rng.integers(0, n, size=n, endpoint=False)
        y_sample = y_true_arr[sample_idx]
        class10_prior_sample = float(np.mean((y_sample == int(class10_id)).astype(np.float32)))
        base_metrics_sample = _compute_candidate_metrics(
            y_true=y_sample,
            y_pred=cached_pred["raw"][sample_idx],
            class10_id=int(class10_id),
            class10_prior=class10_prior_sample,
        )

        sampled_global: Optional[Dict[str, Any]] = None
        sampled_per_class: Optional[Dict[str, Any]] = None
        sampled_nofault_bias: Optional[Dict[str, Any]] = None
        if "global" in cached_pred and isinstance(global_candidate, dict):
            global_metrics_sample = _compute_candidate_metrics(
                y_true=y_sample,
                y_pred=cached_pred["global"][sample_idx],
                class10_id=int(class10_id),
                class10_prior=class10_prior_sample,
            )
            sampled_global = {
                **global_candidate,
                "accuracy_after_threshold": float(global_metrics_sample["accuracy"]),
                "macro_f1_after_threshold": float(global_metrics_sample["macro_f1"]),
                "macro_recall_after_threshold": float(global_metrics_sample["macro_recall"]),
                "gmean_after_threshold": float(global_metrics_sample["gmean"]),
                "pred_rate_class10_after": float(global_metrics_sample["pred_rate_class10"]),
                "class10_overpredict_after": float(global_metrics_sample["class10_overpredict"]),
                "threshold_constraint_flags": _build_threshold_constraint_flags(
                    baseline_metrics=base_metrics_sample,
                    candidate_metrics=global_metrics_sample,
                    min_macro_recall_delta=float(min_macro_recall_delta),
                    min_gmean_delta=float(min_gmean_delta),
                ),
            }
        if "per_class" in cached_pred and isinstance(per_class_candidate, dict):
            per_class_metrics_sample = _compute_candidate_metrics(
                y_true=y_sample,
                y_pred=cached_pred["per_class"][sample_idx],
                class10_id=int(class10_id),
                class10_prior=class10_prior_sample,
            )
            sampled_per_class = {
                **per_class_candidate,
                "accuracy_after_threshold": float(per_class_metrics_sample["accuracy"]),
                "macro_f1_after_threshold": float(per_class_metrics_sample["macro_f1"]),
                "macro_recall_after_threshold": float(per_class_metrics_sample["macro_recall"]),
                "gmean_after_threshold": float(per_class_metrics_sample["gmean"]),
                "pred_rate_class10_after": float(per_class_metrics_sample["pred_rate_class10"]),
                "class10_overpredict_after": float(per_class_metrics_sample["class10_overpredict"]),
                "threshold_constraint_flags": _build_threshold_constraint_flags(
                    baseline_metrics=base_metrics_sample,
                    candidate_metrics=per_class_metrics_sample,
                    min_macro_recall_delta=float(min_macro_recall_delta),
                    min_gmean_delta=float(min_gmean_delta),
                ),
            }
        if "nofault_bias" in cached_pred and isinstance(nofault_bias_candidate, dict):
            nofault_metrics_sample = _compute_candidate_metrics(
                y_true=y_sample,
                y_pred=cached_pred["nofault_bias"][sample_idx],
                class10_id=int(class10_id),
                class10_prior=class10_prior_sample,
            )
            sampled_nofault_bias = {
                **nofault_bias_candidate,
                "accuracy_after_threshold": float(nofault_metrics_sample["accuracy"]),
                "macro_f1_after_threshold": float(nofault_metrics_sample["macro_f1"]),
                "macro_recall_after_threshold": float(nofault_metrics_sample["macro_recall"]),
                "gmean_after_threshold": float(nofault_metrics_sample["gmean"]),
                "pred_rate_class10_after": float(nofault_metrics_sample["pred_rate_class10"]),
                "class10_overpredict_after": float(nofault_metrics_sample["class10_overpredict"]),
                "threshold_constraint_flags": _build_threshold_constraint_flags(
                    baseline_metrics=base_metrics_sample,
                    candidate_metrics=nofault_metrics_sample,
                    min_macro_recall_delta=float(min_macro_recall_delta),
                    min_gmean_delta=float(min_gmean_delta),
                ),
            }

        picked = _select_threshold_candidate(
            baseline_macro_f1=float(base_metrics_sample["macro_f1"]),
            baseline_macro_recall=float(base_metrics_sample["macro_recall"]),
            baseline_gmean=float(base_metrics_sample["gmean"]),
            class10_constraint=str(class10_constraint),
            require_non_decreasing_macro_f1=bool(require_non_decreasing_macro_f1),
            threshold_search_objective=str(threshold_search_objective),
            threshold_search_min_accuracy=float(threshold_search_min_accuracy),
            min_macro_recall_delta=float(min_macro_recall_delta),
            min_gmean_delta=float(min_gmean_delta),
            accuracy_tie_eps=float(accuracy_tie_eps),
            global_candidate=sampled_global,
            per_class_candidate=sampled_per_class,
            nofault_bias_candidate=sampled_nofault_bias,
        )
        picked_mode = str(picked.get("selected_mode", "none"))
        if picked_mode not in mode_counts:
            picked_mode = "none"
        mode_counts[picked_mode] += 1

    selected_mode_key = str(selected_mode) if str(selected_mode) in mode_counts else "none"
    selection_stability_score = float(mode_counts[selected_mode_key] / float(rounds_int))
    ranked_modes = sorted(mode_counts.items(), key=lambda item: item[1], reverse=True)
    winner_mode = str(ranked_modes[0][0]) if len(ranked_modes) > 0 else "none"
    return {
        "rounds": rounds_int,
        "seed": int(seed),
        "mode_counts": mode_counts,
        "selected_mode": selected_mode_key,
        "winner_mode": winner_mode,
        "selection_stability_score": selection_stability_score,
        "baseline_metrics": {
            "macro_f1": float(baseline_metrics.get("macro_f1", 0.0)),
            "macro_recall": float(baseline_metrics.get("macro_recall", 0.0)),
            "gmean": float(baseline_metrics.get("gmean", 0.0)),
            "accuracy": float(baseline_metrics.get("accuracy", 0.0)),
        },
    }


def search_best_global_confidence_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fallback_class_id: int,
    num_thresholds: int = 101,
) -> Dict[str, float]:
    """
    在验证集上搜索全局阈值，目标是最大化 macro-F1。
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if y_true.size == 0:
        return {
            "best_threshold": 0.0,
            "macro_f1_before_threshold": 0.0,
            "macro_f1_after_threshold": 0.0,
        }

    base_pred = np.argmax(y_prob, axis=1).astype(np.int64)
    base_f1 = float(f1_score(y_true, base_pred, average="macro", zero_division=0))

    best_threshold = 0.0
    best_f1 = base_f1
    for threshold in np.linspace(0.0, 1.0, int(max(num_thresholds, 2))):
        pred = _apply_global_confidence_threshold(
            y_prob=y_prob,
            threshold=float(threshold),
            fallback_class_id=int(fallback_class_id),
        )
        score = float(f1_score(y_true, pred, average="macro", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return {
        "best_threshold": float(best_threshold),
        "macro_f1_before_threshold": float(base_f1),
        "macro_f1_after_threshold": float(best_f1),
    }


def _build_eval_dataloader(
    test_records,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    """
    构建评估数据加载器（source/target 共用）。

    参数:
        test_records: 测试记录列表
        mean: z-score 均值
        std: z-score 标准差
        device: 设备
        num_workers: 数据加载线程数

    返回:
        DataLoader 实例
    """
    loader = MissionLoader(max_cache_items=32)

    test_dataset = DeterministicWindowDataset(
        records=test_records,
        zscore_mean=mean,
        zscore_std=std,
        loader=loader,
        windows_per_scale=CFG.eval_windows_per_scale,
        base_seed=CFG.split_seed,
    )

    dataloader_kwargs = {
        "batch_size": CFG.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_padded,
        "drop_last": False,
    }

    if bool(CFG.pin_memory and device.type == "cuda"):
        dataloader_kwargs["pin_memory"] = True
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = CFG.persistent_workers
        dataloader_kwargs["prefetch_factor"] = CFG.prefetch_factor

    return DataLoader(test_dataset, **dataloader_kwargs)


def _apply_adabn_statistics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 0,
) -> tuple[int, int]:
    """
    仅更新 BatchNorm 统计量（AdaBN），不更新任何权重。

    返回:
        (实际执行的 batch 数, BN 层数量)
    """
    bn_layers = [module for module in model.modules() if isinstance(module, _BatchNorm)]
    if len(bn_layers) == 0:
        model.eval()
        return 0, 0

    # 关闭 Dropout，仅打开 BN 的 train 状态以更新 running stats。
    model.eval()
    for module in bn_layers:
        module.train()

    amp_enabled = bool(CFG.use_amp and device.type == "cuda")
    non_blocking = bool(device.type == "cuda" and CFG.pin_memory)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    updated_batches = 0
    with torch.no_grad():
        for x, lengths, _ in dataloader:
            x = x.to(device, non_blocking=non_blocking)
            lengths = lengths.to(device, non_blocking=non_blocking)
            with torch.autocast(
                device_type=autocast_device,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                _ = model(x, lengths)
            updated_batches += 1
            if max_batches > 0 and updated_batches >= max_batches:
                break

    model.eval()
    return updated_batches, len(bn_layers)


def _collect_split_labels_and_probabilities(
    run_dir: Path,
    checkpoint: str,
    domain: str,
    split: str,
    probability_temperature: float = 1.0,
    return_raw_logits: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """
    收集指定分割上的真实标签与预测概率。
    """
    if float(probability_temperature) <= 0.0:
        raise ValueError(f"probability_temperature 必须 > 0，当前={probability_temperature}")
    device = resolve_device(CFG.device)
    num_workers = resolve_num_workers(CFG.num_workers)
    records, mean, std = _load_domain_artifacts(run_dir, domain=domain, split=split)
    if len(records) == 0:
        empty_labels = np.array([], dtype=np.int64)
        empty_prob = np.empty((0, CFG.num_classes), dtype=np.float32)
        if bool(return_raw_logits):
            empty_logits = np.empty((0, CFG.num_classes), dtype=np.float32)
            return empty_labels, empty_prob, empty_logits
        return empty_labels, empty_prob

    dataloader = _build_eval_dataloader(
        records,
        mean,
        std,
        device=device,
        num_workers=num_workers,
    )

    model, _, load_meta = load_temporal_convnet_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
    )
    if load_meta["model_kwargs_source"] == "inferred_from_state_dict":
        print(
            "[WARN] checkpoint 缺少 model_init_kwargs，已基于 state_dict 推断结构加载（可能受配置漂移影响）。"
        )
    model.eval()

    amp_enabled = bool(CFG.use_amp and device.type == "cuda")
    non_blocking = bool(device.type == "cuda" and CFG.pin_memory)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    with torch.no_grad():
        for x, lengths, y in dataloader:
            x = x.to(device, non_blocking=non_blocking)
            lengths = lengths.to(device, non_blocking=non_blocking)
            with torch.autocast(
                device_type=autocast_device,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                logits, _ = model.forward_with_aux(x, lengths)
            raw_logits = logits.detach().cpu().numpy().astype(np.float32)
            logits_for_prob = logits
            if abs(float(probability_temperature) - 1.0) > 1e-12:
                logits_for_prob = logits_for_prob / float(probability_temperature)
            probs = torch.softmax(logits_for_prob, dim=-1).cpu().numpy().astype(np.float32)
            all_probs.append(probs)
            all_labels.append(y.numpy().astype(np.int64))
            if bool(return_raw_logits):
                all_logits.append(raw_logits)

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    if bool(return_raw_logits):
        y_logits = np.concatenate(all_logits, axis=0)
        return y_true, y_prob, y_logits
    return y_true, y_prob


def _extract_domain_features(
    run_dir: Path,
    checkpoint: str,
    domain: str,
    max_samples: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    提取指定域的特征（分类头前的 pooled 特征）。

    参数:
        run_dir: 运行目录
        checkpoint: 模型 checkpoint 路径
        domain: "source" 或 "target"
        max_samples: 最大采样数（可选）

    返回:
        (features, labels) 元组
        features: [N, D] 特征矩阵
        labels: [N] 标签数组
    """
    device = resolve_device(CFG.device)
    num_workers = resolve_num_workers(CFG.num_workers)

    # 加载数据
    test_records, mean, std = _load_domain_artifacts(run_dir, domain)

    if len(test_records) == 0:
        return np.array([]), np.array([])

    dataloader = _build_eval_dataloader(
        test_records, mean, std, device=device, num_workers=num_workers
    )

    # 加载模型
    model, _, load_meta = load_temporal_convnet_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
    )
    if load_meta["model_kwargs_source"] == "inferred_from_state_dict":
        print(
            "[WARN] checkpoint 缺少 model_init_kwargs，已基于 state_dict 推断结构加载（可能受配置漂移影响）。"
        )
    model.eval()

    amp_enabled = bool(CFG.use_amp and device.type == "cuda")
    non_blocking = bool(device.type == "cuda" and CFG.pin_memory)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, lengths, y in dataloader:
            x = x.to(device, non_blocking=non_blocking)
            lengths = lengths.to(device, non_blocking=non_blocking)

            with torch.autocast(
                device_type=autocast_device,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                features = model.get_features(x, lengths)

            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    # 可选采样
    if max_samples is not None and len(features) > max_samples:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(features), size=max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    return features, labels


def _accumulate_attention_summary(
    summary: Dict[str, Dict[str, Any]],
    labels: torch.Tensor,
    lengths: torch.Tensor,
    time_attn_weights: Optional[torch.Tensor],
    channel_attn_weights: Optional[torch.Tensor],
) -> None:
    """
    将一个 batch 的注意力信息累积到类别级摘要。
    """

    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    lengths_np = lengths.detach().cpu().numpy().astype(np.int64)
    time_np = None if time_attn_weights is None else time_attn_weights.detach().cpu().numpy()
    channel_np = None if channel_attn_weights is None else channel_attn_weights.detach().cpu().numpy()

    for i, class_id in enumerate(labels_np.tolist()):
        key = str(int(class_id))
        if key not in summary:
            summary[key] = {
                "count": 0,
                "time_peak_index_sum": 0.0,
                "time_peak_ratio_sum": 0.0,
                "channel_weight_sum": None,
            }
        bucket = summary[key]
        bucket["count"] += 1

        valid_len = int(max(lengths_np[i], 0))
        if time_np is not None and valid_len > 0:
            time_valid = time_np[i, :valid_len]
            peak_idx = int(np.argmax(time_valid))
            bucket["time_peak_index_sum"] += float(peak_idx)
            bucket["time_peak_ratio_sum"] += float(peak_idx / max(valid_len - 1, 1))

        if channel_np is not None:
            vec = channel_np[i].astype(np.float64)
            if bucket["channel_weight_sum"] is None:
                bucket["channel_weight_sum"] = vec
            else:
                bucket["channel_weight_sum"] += vec


def _finalize_attention_summary(summary: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    将累积统计量转换为可落盘的类别摘要。
    """

    out: Dict[str, Dict[str, Any]] = {}
    for class_id, bucket in summary.items():
        count = int(bucket["count"])
        if count <= 0:
            continue
        item: Dict[str, Any] = {
            "count": count,
            "time_peak_index_mean": float(bucket["time_peak_index_sum"] / count),
            "time_peak_ratio_mean": float(bucket["time_peak_ratio_sum"] / count),
        }
        if bucket["channel_weight_sum"] is not None:
            item["channel_weight_mean"] = (
                (bucket["channel_weight_sum"] / count).astype(np.float32).tolist()
            )
        else:
            item["channel_weight_mean"] = []
        out[class_id] = item
    return out


def evaluate_model(
    run_dir: Path,
    checkpoint: str,
    domain: str = "source",
    model_name: str = "tcn",
    split: str = "test",
    global_threshold: Optional[float] = None,
    per_class_thresholds: Optional[List[float]] = None,
    nofault_logit_bias: Optional[float] = None,
    threshold_fallback_class_id: Optional[int] = None,
    threshold_mode: Optional[str] = None,
    threshold_applied: Optional[bool] = None,
    threshold_selection_reason: Optional[str] = None,
    probability_temperature: float = 1.0,
) -> EvalResult:
    """
    在测试集上评估模型
    
    参数:
        run_dir: 运行目录
        checkpoint: 模型 checkpoint 路径
        domain: "source" 或 "target"
        model_name: 模型名称（用于日志）
        split: 评估分割（test/val）
        global_threshold: 全局置信度阈值（可选）
        per_class_thresholds: 每类阈值（可选）
        nofault_logit_bias: no_fault 类别的 logit 负偏移（可选）
        threshold_fallback_class_id: 阈值回退类别 ID（可选）
        threshold_mode: 阈值模式标记（可选）
        threshold_applied: 是否应用阈值（可选）
        threshold_selection_reason: 阈值选择原因（可选）
        probability_temperature: softmax 前的 logit 温度缩放（>0）
    
    返回:
        EvalResult
    """
    device = resolve_device(CFG.device)
    num_workers = resolve_num_workers(CFG.num_workers)
    strategy_count = int(global_threshold is not None) + int(per_class_thresholds is not None) + int(
        nofault_logit_bias is not None
    )
    if strategy_count > 1:
        raise ValueError(
            "global_threshold/per_class_thresholds/nofault_logit_bias 不能同时设置多个"
        )
    if nofault_logit_bias is not None and float(nofault_logit_bias) < 0.0:
        raise ValueError(f"nofault_logit_bias 必须 >= 0，当前={nofault_logit_bias}")
    if float(probability_temperature) <= 0.0:
        raise ValueError(f"probability_temperature 必须 > 0，当前={probability_temperature}")
    
    # 加载数据
    test_records, mean, std = _load_domain_artifacts(run_dir, domain, split=split)
    
    if len(test_records) == 0:
        print(f"[评估] {domain} 测试集为空")
        return EvalResult(
            model_name=model_name,
            domain=domain,
            split=split,
            accuracy=0.0,
            macro_f1=0.0,
            macro_recall=0.0,
            weighted_f1=0.0,
            gmean=0.0,
            confusion_matrix=[],
            per_class_metrics=[],
            top_confusion_pairs=[],
            attention_summary={},
            global_threshold=global_threshold,
            macro_f1_before_threshold=None,
            threshold_mode=threshold_mode,
            per_class_thresholds=(
                None if per_class_thresholds is None else [float(v) for v in per_class_thresholds]
            ),
            threshold_applied=threshold_applied,
            threshold_selection_reason=threshold_selection_reason,
            probability_temperature=float(probability_temperature),
            nofault_logit_bias=nofault_logit_bias,
        )
    
    loader = MissionLoader(max_cache_items=32)
    
    test_dataset = DeterministicWindowDataset(
        records=test_records,
        zscore_mean=mean,
        zscore_std=std,
        loader=loader,
        windows_per_scale=CFG.eval_windows_per_scale,
        base_seed=CFG.split_seed,
    )

    dataloader_runtime_kwargs = {}
    if bool(CFG.pin_memory and device.type == "cuda"):
        dataloader_runtime_kwargs["pin_memory"] = True
    if num_workers > 0:
        dataloader_runtime_kwargs["persistent_workers"] = CFG.persistent_workers
        dataloader_runtime_kwargs["prefetch_factor"] = CFG.prefetch_factor

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_padded,
        drop_last=False,
        **dataloader_runtime_kwargs,
    )
    
    # 加载模型
    print(f"[评估] 加载模型: {checkpoint}")
    model, _, load_meta = load_temporal_convnet_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
    )
    if load_meta["model_kwargs_source"] == "inferred_from_state_dict":
        print(
            "[WARN] checkpoint 缺少 model_init_kwargs，已基于 state_dict 推断结构加载（可能受配置漂移影响）。"
        )
    if bool(getattr(CFG, "ft_adabn_enable", False)) and domain == "target":
        adabn_split = str(getattr(CFG, "ft_adabn_split", "val")).strip().lower()
        adabn_max_batches = int(getattr(CFG, "ft_adabn_max_batches", 0))
        try:
            adabn_records, _, _ = _load_domain_artifacts(
                run_dir=run_dir,
                domain="target",
                split=adabn_split,
            )
            if len(adabn_records) > 0:
                adabn_loader = _build_eval_dataloader(
                    adabn_records,
                    mean,
                    std,
                    device=device,
                    num_workers=num_workers,
                )
                updated_batches, bn_layer_count = _apply_adabn_statistics(
                    model=model,
                    dataloader=adabn_loader,
                    device=device,
                    max_batches=adabn_max_batches,
                )
                print(
                    f"[AdaBN] split=target_{adabn_split} "
                    f"bn_layers={bn_layer_count} updated_batches={updated_batches}"
                )
            else:
                print(f"[AdaBN] split=target_{adabn_split} 样本为空，已跳过")
        except Exception as exc:
            print(f"[WARN] AdaBN 执行失败，继续常规评估: {exc}")
    model.eval()
    amp_enabled = bool(CFG.use_amp and device.type == "cuda")
    non_blocking = bool(device.type == "cuda" and CFG.pin_memory)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    # 推理
    all_preds = []
    all_raw_preds = []
    all_labels = []
    attention_summary_raw: Dict[str, Dict[str, Any]] = {}

    with torch.no_grad():
        for x, lengths, y in test_loader:
            x = x.to(device, non_blocking=non_blocking)
            lengths = lengths.to(device, non_blocking=non_blocking)
            with torch.autocast(
                device_type=autocast_device,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                logits, aux = model.forward_with_aux(x, lengths)

            logits_for_raw_prob = logits
            if abs(float(probability_temperature) - 1.0) > 1e-12:
                logits_for_raw_prob = logits_for_raw_prob / float(probability_temperature)
            raw_probs = torch.softmax(logits_for_raw_prob, dim=-1).cpu().numpy()
            raw_preds = np.argmax(raw_probs, axis=1).astype(np.int64)

            if nofault_logit_bias is not None:
                class10_id = int(getattr(CFG, "fault_to_class", {}).get(10, CFG.num_classes - 1))
                logits_for_prob = logits.clone()
                logits_for_prob[:, class10_id] = (
                    logits_for_prob[:, class10_id] - float(nofault_logit_bias)
                )
                if abs(float(probability_temperature) - 1.0) > 1e-12:
                    logits_for_prob = logits_for_prob / float(probability_temperature)
                probs = torch.softmax(logits_for_prob, dim=-1).cpu().numpy()
            else:
                probs = raw_probs

            if global_threshold is not None:
                fallback_class_id = (
                    int(threshold_fallback_class_id)
                    if threshold_fallback_class_id is not None
                    else int(getattr(CFG, "fault_to_class", {}).get(10, CFG.num_classes - 1))
                )
                preds = _apply_global_confidence_threshold(
                    y_prob=probs,
                    threshold=float(global_threshold),
                    fallback_class_id=fallback_class_id,
                )
            elif per_class_thresholds is not None:
                preds = _apply_per_class_thresholds(
                    y_prob=probs,
                    thresholds=np.asarray(per_class_thresholds, dtype=np.float32),
                )
            elif nofault_logit_bias is not None:
                preds = np.argmax(probs, axis=1).astype(np.int64)
            else:
                preds = raw_preds
            all_raw_preds.extend(raw_preds.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
            _accumulate_attention_summary(
                summary=attention_summary_raw,
                labels=y,
                lengths=lengths.cpu(),
                time_attn_weights=aux.get("time_attn_weights"),  # type: ignore[arg-type]
                channel_attn_weights=aux.get("channel_attn_weights_summary"),  # type: ignore[arg-type]
            )
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_pred_raw = np.array(all_raw_preds)
    
    # 计算指标
    metrics = compute_metrics(y_true, y_pred, num_classes=CFG.num_classes)
    gmean = compute_gmean(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=CFG.num_classes)
    attention_summary = _finalize_attention_summary(attention_summary_raw)
    class_names = _build_class_names_from_config()
    per_class_metrics = _compute_per_class_metrics(cm=cm, class_names=class_names)
    top_confusion_pairs = _compute_top_confusion_pairs(
        cm=cm,
        class_names=class_names,
        top_k=10,
    )
    
    print(f"[评估] {domain} 域 | model={model_name}")
    if (
        global_threshold is not None
        or per_class_thresholds is not None
        or nofault_logit_bias is not None
    ):
        macro_f1_before_threshold = float(
            f1_score(y_true, y_pred_raw, average="macro", zero_division=0)
        )
        if global_threshold is not None:
            print(
                f"  Threshold Search(global): threshold={float(global_threshold):.4f} "
                f"macro_f1_before={macro_f1_before_threshold:.4f}"
            )
        else:
            if per_class_thresholds is not None:
                print(
                    f"  Threshold Search(per_class): macro_f1_before={macro_f1_before_threshold:.4f}"
                )
            else:
                print(
                    "  Threshold Search(nofault_bias): "
                    f"bias={float(nofault_logit_bias):.4f} "
                    f"macro_f1_before={macro_f1_before_threshold:.4f}"
                )
    else:
        macro_f1_before_threshold = None
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  Macro-Recall: {metrics['macro_recall']:.4f}")
    print(f"  Weighted-F1: {metrics['weighted_f1']:.4f}")
    # G-mean 可能非常小（当某些类别召回率为 0 时），用科学计数法更便于排查与对比
    print(f"  G-mean: {gmean:.4e}")
    print(f"  Attention Summary Classes: {len(attention_summary)}")
    
    # 保存结果
    result = EvalResult(
        model_name=model_name,
        domain=domain,
        split=split,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        macro_recall=metrics["macro_recall"],
        weighted_f1=metrics["weighted_f1"],
        gmean=gmean,
        confusion_matrix=cm.tolist(),
        per_class_metrics=per_class_metrics,
        top_confusion_pairs=top_confusion_pairs,
        attention_summary=attention_summary,
        global_threshold=global_threshold,
        macro_f1_before_threshold=macro_f1_before_threshold,
        threshold_mode=threshold_mode,
        per_class_thresholds=(
            None if per_class_thresholds is None else [float(v) for v in per_class_thresholds]
        ),
        threshold_applied=(
            bool(
                global_threshold is not None
                or per_class_thresholds is not None
                or nofault_logit_bias is not None
            )
            if threshold_applied is None
            else bool(threshold_applied)
        ),
        threshold_selection_reason=threshold_selection_reason,
        probability_temperature=float(probability_temperature),
        nofault_logit_bias=nofault_logit_bias,
    )
    
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = eval_dir / f"{model_name}_{domain}.json"
    write_json(result_path, {
        "model_name": result.model_name,
        "domain": result.domain,
        "split": result.split,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "macro_recall": result.macro_recall,
        "weighted_f1": result.weighted_f1,
        "gmean": result.gmean,
        "confusion_matrix": result.confusion_matrix,
        "per_class_metrics": result.per_class_metrics,
        "top_confusion_pairs": result.top_confusion_pairs,
        "attention_summary": result.attention_summary,
        "global_threshold": result.global_threshold,
        "macro_f1_before_threshold": result.macro_f1_before_threshold,
        "threshold_mode": result.threshold_mode,
        "per_class_thresholds": result.per_class_thresholds,
        "threshold_applied": result.threshold_applied,
        "threshold_selection_reason": result.threshold_selection_reason,
        "probability_temperature": result.probability_temperature,
        "nofault_logit_bias": result.nofault_logit_bias,
    })

    # 在评估完成后直接导出热力图，确保每次测试都具备可视化产物。
    heatmap_path = eval_dir / f"cm_{model_name}_{domain}.png"
    saved_heatmap = _save_confusion_matrix_heatmap(
        cm=cm,
        output_path=heatmap_path,
        title=f"Confusion Matrix | {model_name} | {domain}",
        class_names=class_names,
    )
    
    print(f"  结果已保存: {result_path}")
    if saved_heatmap is not None:
        print(f"  混淆矩阵热力图已保存: {saved_heatmap}")
    
    return result


def evaluate_transfer_gap(
    run_dir: Path,
    source_ckpt: str,
    target_ckpt: str,
) -> Dict:
    """
    评估迁移学习效果

    参数:
        run_dir: 运行目录
        source_ckpt: 源域模型 checkpoint
        target_ckpt: 微调后模型 checkpoint

    返回:
        比较结果
    """
    print("\n" + "=" * 60)
    print("迁移学习效果评估")
    print("=" * 60)

    # 源域模型在源域测试集的表现
    print("\n[1] 源域模型 → 源域测试集")
    source_on_source = evaluate_model(
        run_dir, source_ckpt, domain="source", model_name="source_model"
    )

    # 源域模型在目标域测试集的表现（直接迁移）
    print("\n[2] 源域模型 → 目标域测试集（直接迁移，无微调）")
    source_on_target = evaluate_model(
        run_dir, source_ckpt, domain="target", model_name="source_model_direct"
    )

    threshold_search_enabled = bool(getattr(CFG, "threshold_search", False))
    threshold_search_mode = str(getattr(CFG, "threshold_search_mode", "auto")).strip().lower()
    threshold_search_nofault_bias = bool(
        getattr(CFG, "threshold_search_nofault_bias", True)
    )
    threshold_search_max_accuracy_drop = float(
        getattr(CFG, "threshold_search_max_accuracy_drop", 0.0)
    )
    threshold_search_class10_constraint = str(
        getattr(CFG, "threshold_search_class10_constraint", "soft")
    ).strip().lower()
    threshold_search_require_non_decreasing_macro_f1 = bool(
        getattr(CFG, "threshold_search_require_non_decreasing_macro_f1", True)
    )
    threshold_search_class10_overpredict_limit = float(
        getattr(CFG, "threshold_search_class10_overpredict_limit", 0.0)
    )
    threshold_search_min_macro_recall_delta = float(
        getattr(CFG, "threshold_search_min_macro_recall_delta", -0.01)
    )
    threshold_search_min_gmean_delta = float(
        getattr(CFG, "threshold_search_min_gmean_delta", -0.02)
    )
    threshold_search_accuracy_tie_eps = float(
        getattr(CFG, "threshold_search_accuracy_tie_eps", 0.0015)
    )
    threshold_search_bootstrap_rounds = int(
        getattr(CFG, "threshold_search_bootstrap_rounds", 200)
    )
    threshold_search_objective = _normalize_threshold_search_objective(
        str(
            getattr(
                CFG,
                "threshold_search_objective",
                THRESHOLD_OBJECTIVE_F1_WITH_ACC_FLOOR,
            )
        )
    )
    threshold_search_min_accuracy = float(
        getattr(CFG, "threshold_search_min_accuracy", 0.900)
    )
    threshold_temperature = float(getattr(CFG, "threshold_temperature", 1.0))
    fallback_class_id = int(getattr(CFG, "fault_to_class", {}).get(10, CFG.num_classes - 1))
    class10_id = int(getattr(CFG, "fault_to_class", {}).get(10, CFG.num_classes - 1))
    objective_payload_text = _objective_to_payload_text(
        threshold_search_objective,
        threshold_search_min_accuracy,
    )
    threshold_search_result: Dict[str, Any] = {
        "schema_version": 2,
        "objective": objective_payload_text,
        "enabled": threshold_search_enabled,
        "mode": threshold_search_mode,
        "nofault_bias_enabled": threshold_search_nofault_bias,
        "threshold_search_objective": threshold_search_objective,
        "threshold_search_min_accuracy": threshold_search_min_accuracy,
        "max_accuracy_drop": threshold_search_max_accuracy_drop,
        "class10_constraint": threshold_search_class10_constraint,
        "require_non_decreasing_macro_f1": threshold_search_require_non_decreasing_macro_f1,
        "class10_overpredict_limit": threshold_search_class10_overpredict_limit,
        "min_macro_recall_delta": threshold_search_min_macro_recall_delta,
        "min_gmean_delta": threshold_search_min_gmean_delta,
        "accuracy_tie_eps": threshold_search_accuracy_tie_eps,
        "bootstrap_rounds": threshold_search_bootstrap_rounds,
        "temperature": threshold_temperature,
        "val_metrics": {
            "objective": objective_payload_text,
            "threshold_search_objective": threshold_search_objective,
            "threshold_search_min_accuracy": threshold_search_min_accuracy,
            "require_non_decreasing_macro_f1": threshold_search_require_non_decreasing_macro_f1,
            "class10_overpredict_limit": threshold_search_class10_overpredict_limit,
            "min_macro_recall_delta": threshold_search_min_macro_recall_delta,
            "min_gmean_delta": threshold_search_min_gmean_delta,
            "accuracy_tie_eps": threshold_search_accuracy_tie_eps,
            "baseline": None,
            "global_candidate": None,
            "per_class_candidate": None,
            "nofault_bias_candidate": None,
            "selected_mode": "none",
            "selected_candidate": None,
            "selection_stability": None,
            "threshold_constraint_flags": None,
            "selection_reason": (
                "threshold_search_disabled"
                if not threshold_search_enabled
                else "pending"
            ),
        },
        "test_metrics": None,
    }
    val_metrics = threshold_search_result["val_metrics"]
    if threshold_search_enabled:
        print("\n[阈值搜索] 基于 target val 进行阈值搜索")
        try:
            val_collected = _collect_split_labels_and_probabilities(
                run_dir=run_dir,
                checkpoint=target_ckpt,
                domain="target",
                split="val",
                probability_temperature=threshold_temperature,
                return_raw_logits=threshold_search_nofault_bias,
            )
            if threshold_search_nofault_bias:
                y_val, y_prob_val, y_val_raw_logits = val_collected
            else:
                y_val, y_prob_val = val_collected
                y_val_raw_logits = None
            if y_val.size == 0:
                val_metrics["selection_reason"] = "target_val_empty"
            else:
                class10_prior = float(np.mean((y_val == class10_id).astype(np.float32)))
                y_val_raw_pred = np.argmax(y_prob_val, axis=1).astype(np.int64)
                baseline_metrics = _compute_candidate_metrics(
                    y_val,
                    y_val_raw_pred,
                    class10_id=class10_id,
                    class10_prior=class10_prior,
                )
                val_metrics["baseline"] = {
                    "macro_f1": float(baseline_metrics["macro_f1"]),
                    "macro_recall": float(baseline_metrics["macro_recall"]),
                    "gmean": float(baseline_metrics["gmean"]),
                    "accuracy": float(baseline_metrics["accuracy"]),
                    "pred_rate_class10": float(baseline_metrics["pred_rate_class10"]),
                    "class10_prior": float(class10_prior),
                }

                global_candidate: Optional[Dict[str, Any]] = None
                per_class_candidate: Optional[Dict[str, Any]] = None
                nofault_bias_candidate: Optional[Dict[str, Any]] = None

                if threshold_search_mode in {"auto", "global"}:
                    global_candidate = search_best_global_confidence_threshold_candidate(
                        y_true=y_val,
                        y_prob=y_prob_val,
                        fallback_class_id=fallback_class_id,
                        class10_id=class10_id,
                        class10_prior=class10_prior,
                        max_accuracy_drop=threshold_search_max_accuracy_drop,
                        class10_constraint=threshold_search_class10_constraint,
                        class10_overpredict_limit=threshold_search_class10_overpredict_limit,
                        min_macro_recall_delta=threshold_search_min_macro_recall_delta,
                        min_gmean_delta=threshold_search_min_gmean_delta,
                        threshold_search_objective=threshold_search_objective,
                        threshold_search_min_accuracy=threshold_search_min_accuracy,
                        accuracy_tie_eps=threshold_search_accuracy_tie_eps,
                    )
                    val_metrics["global_candidate"] = global_candidate
                    if bool(global_candidate.get("is_valid", False)):
                        print(
                            f"  [global] threshold={float(global_candidate['best_threshold']):.4f} "
                            f"val_macro_f1={float(global_candidate['macro_f1_after_threshold']):.4f} "
                            f"val_acc={float(global_candidate['accuracy_after_threshold']):.4f}"
                        )
                    else:
                        print(
                            f"  [global] 无可行候选: {global_candidate.get('selection_reason')}"
                        )

                if threshold_search_mode in {"auto", "per_class"}:
                    per_class_candidate = search_best_per_class_confidence_threshold(
                        y_true=y_val,
                        y_prob=y_prob_val,
                        class10_id=class10_id,
                        class10_prior=class10_prior,
                        max_accuracy_drop=threshold_search_max_accuracy_drop,
                        class10_constraint=threshold_search_class10_constraint,
                        class10_overpredict_limit=threshold_search_class10_overpredict_limit,
                        min_macro_recall_delta=threshold_search_min_macro_recall_delta,
                        min_gmean_delta=threshold_search_min_gmean_delta,
                        threshold_search_objective=threshold_search_objective,
                        threshold_search_min_accuracy=threshold_search_min_accuracy,
                        accuracy_tie_eps=threshold_search_accuracy_tie_eps,
                    )
                    val_metrics["per_class_candidate"] = per_class_candidate
                    if bool(per_class_candidate.get("is_valid", False)):
                        print(
                            f"  [per_class] val_macro_f1={float(per_class_candidate['macro_f1_after_threshold']):.4f} "
                            f"val_acc={float(per_class_candidate['accuracy_after_threshold']):.4f}"
                        )
                    else:
                        print(
                            f"  [per_class] 无可行候选: {per_class_candidate.get('selection_reason')}"
                        )

                if (
                    threshold_search_nofault_bias
                    and threshold_search_mode in {"auto", "nofault_bias"}
                ):
                    if y_val_raw_logits is None:
                        val_metrics["nofault_bias_candidate"] = {
                            "mode": "nofault_bias",
                            "is_valid": False,
                            "selection_reason": "missing_raw_logits",
                        }
                    else:
                        nofault_bias_candidate = search_best_nofault_logit_bias(
                            y_true=y_val,
                            raw_logits=y_val_raw_logits,
                            class10_id=class10_id,
                            class10_prior=class10_prior,
                            temperature=threshold_temperature,
                            max_accuracy_drop=threshold_search_max_accuracy_drop,
                            class10_constraint=threshold_search_class10_constraint,
                            class10_overpredict_limit=threshold_search_class10_overpredict_limit,
                            min_macro_recall_delta=threshold_search_min_macro_recall_delta,
                            min_gmean_delta=threshold_search_min_gmean_delta,
                            threshold_search_objective=threshold_search_objective,
                            threshold_search_min_accuracy=threshold_search_min_accuracy,
                            accuracy_tie_eps=threshold_search_accuracy_tie_eps,
                            bias_min=0.0,
                            bias_max=5.0,
                            bias_step=0.1,
                        )
                        val_metrics["nofault_bias_candidate"] = nofault_bias_candidate
                        if bool(nofault_bias_candidate.get("is_valid", False)):
                            print(
                                "  [nofault_bias] "
                                f"bias={float(nofault_bias_candidate['nofault_logit_bias']):.4f} "
                                f"val_macro_f1={float(nofault_bias_candidate['macro_f1_after_threshold']):.4f} "
                                f"val_acc={float(nofault_bias_candidate['accuracy_after_threshold']):.4f}"
                            )
                        else:
                            print(
                                "  [nofault_bias] 无可行候选: "
                                f"{nofault_bias_candidate.get('selection_reason')}"
                            )

                selected = _select_threshold_candidate(
                    baseline_macro_f1=float(baseline_metrics["macro_f1"]),
                    baseline_macro_recall=float(baseline_metrics["macro_recall"]),
                    baseline_gmean=float(baseline_metrics["gmean"]),
                    class10_constraint=threshold_search_class10_constraint,
                    require_non_decreasing_macro_f1=threshold_search_require_non_decreasing_macro_f1,
                    threshold_search_objective=threshold_search_objective,
                    threshold_search_min_accuracy=threshold_search_min_accuracy,
                    min_macro_recall_delta=threshold_search_min_macro_recall_delta,
                    min_gmean_delta=threshold_search_min_gmean_delta,
                    accuracy_tie_eps=threshold_search_accuracy_tie_eps,
                    global_candidate=global_candidate,
                    per_class_candidate=per_class_candidate,
                    nofault_bias_candidate=nofault_bias_candidate,
                )
                val_metrics.update(selected)
                val_metrics["threshold_constraint_flags"] = selected.get(
                    "threshold_constraint_flags"
                )
                if threshold_search_bootstrap_rounds > 0:
                    val_metrics["selection_stability"] = _bootstrap_selection_stability(
                        y_true=y_val,
                        y_prob=y_prob_val,
                        class10_id=class10_id,
                        fallback_class_id=fallback_class_id,
                        class10_constraint=threshold_search_class10_constraint,
                        require_non_decreasing_macro_f1=threshold_search_require_non_decreasing_macro_f1,
                        threshold_search_objective=threshold_search_objective,
                        threshold_search_min_accuracy=threshold_search_min_accuracy,
                        min_macro_recall_delta=threshold_search_min_macro_recall_delta,
                        min_gmean_delta=threshold_search_min_gmean_delta,
                        accuracy_tie_eps=threshold_search_accuracy_tie_eps,
                        baseline_metrics=baseline_metrics,
                        global_candidate=global_candidate,
                        per_class_candidate=per_class_candidate,
                        nofault_bias_candidate=nofault_bias_candidate,
                        selected_mode=str(val_metrics.get("selected_mode", "none")),
                        rounds=threshold_search_bootstrap_rounds,
                        seed=int(getattr(CFG, "run_seed", 42)),
                        probability_temperature=threshold_temperature,
                    )
                if val_metrics["selected_mode"] == "none":
                    print(
                        f"  [selected] none: {val_metrics['selection_reason']}"
                    )
                else:
                    selected_candidate = val_metrics["selected_candidate"]
                    print(
                        f"  [selected] mode={val_metrics['selected_mode']} "
                        f"val_macro_f1={float(selected_candidate['macro_f1_after_threshold']):.4f} "
                        f"val_acc={float(selected_candidate['accuracy_after_threshold']):.4f}"
                    )
                    if isinstance(val_metrics.get("selection_stability"), dict):
                        print(
                            "  [stability] "
                            f"score={float(val_metrics['selection_stability'].get('selection_stability_score', 0.0)):.4f} "
                            f"rounds={int(val_metrics['selection_stability'].get('rounds', 0))}"
                        )
        except Exception as exc:
            print(f"  [警告] 阈值搜索失败，回退为默认评估: {exc}")
            val_metrics["selected_mode"] = "none"
            val_metrics["selected_candidate"] = None
            val_metrics["selection_reason"] = f"search_failed: {exc}"

    # 微调后模型在目标域测试集的表现
    print("\n[3] 微调模型 → 目标域测试集")
    finetuned_on_target = evaluate_model(
        run_dir, target_ckpt, domain="target", model_name="finetuned_model"
    )
    print("\n[3b] 微调模型 → 目标域测试集（阈值策略结果）")
    selected_mode = str(val_metrics.get("selected_mode", "none"))
    selected_candidate = val_metrics.get("selected_candidate")
    if selected_mode == "global" and isinstance(selected_candidate, dict):
        finetuned_on_target_thresholded = evaluate_model(
            run_dir,
            target_ckpt,
            domain="target",
            model_name="finetuned_model_thresholded",
            global_threshold=float(selected_candidate["best_threshold"]),
            threshold_fallback_class_id=fallback_class_id,
            threshold_mode="global",
            threshold_applied=True,
            threshold_selection_reason=str(val_metrics.get("selection_reason")),
            probability_temperature=threshold_temperature,
        )
    elif selected_mode == "per_class" and isinstance(selected_candidate, dict):
        finetuned_on_target_thresholded = evaluate_model(
            run_dir,
            target_ckpt,
            domain="target",
            model_name="finetuned_model_thresholded",
            per_class_thresholds=[
                float(v) for v in selected_candidate.get("per_class_thresholds", [])
            ],
            threshold_mode="per_class",
            threshold_applied=True,
            threshold_selection_reason=str(val_metrics.get("selection_reason")),
            probability_temperature=threshold_temperature,
        )
    elif selected_mode == "nofault_bias" and isinstance(selected_candidate, dict):
        finetuned_on_target_thresholded = evaluate_model(
            run_dir,
            target_ckpt,
            domain="target",
            model_name="finetuned_model_thresholded",
            nofault_logit_bias=float(selected_candidate.get("nofault_logit_bias", 0.0)),
            threshold_mode="nofault_bias",
            threshold_applied=True,
            threshold_selection_reason=str(val_metrics.get("selection_reason")),
            probability_temperature=threshold_temperature,
        )
    else:
        finetuned_on_target_thresholded = evaluate_model(
            run_dir,
            target_ckpt,
            domain="target",
            model_name="finetuned_model_thresholded",
            threshold_mode="none",
            threshold_applied=False,
            threshold_selection_reason=str(val_metrics.get("selection_reason")),
        )

    # 计算改进
    f1_improvement = finetuned_on_target.macro_f1 - source_on_target.macro_f1
    f1_improvement_thresholded = (
        finetuned_on_target_thresholded.macro_f1 - source_on_target.macro_f1
    )
    threshold_search_result["test_metrics"] = {
        "objective": objective_payload_text,
        "threshold_search_objective": threshold_search_objective,
        "threshold_search_min_accuracy": threshold_search_min_accuracy,
        "require_non_decreasing_macro_f1": threshold_search_require_non_decreasing_macro_f1,
        "class10_overpredict_limit": threshold_search_class10_overpredict_limit,
        "min_macro_recall_delta": threshold_search_min_macro_recall_delta,
        "min_gmean_delta": threshold_search_min_gmean_delta,
        "accuracy_tie_eps": threshold_search_accuracy_tie_eps,
        "bootstrap_rounds": threshold_search_bootstrap_rounds,
        "raw": _build_eval_metrics_payload(finetuned_on_target),
        "thresholded": _build_eval_metrics_payload(finetuned_on_target_thresholded),
        "delta_thresholded_minus_raw": {
            "accuracy": float(
                finetuned_on_target_thresholded.accuracy - finetuned_on_target.accuracy
            ),
            "macro_f1": float(
                finetuned_on_target_thresholded.macro_f1 - finetuned_on_target.macro_f1
            ),
            "macro_recall": float(
                finetuned_on_target_thresholded.macro_recall
                - finetuned_on_target.macro_recall
            ),
            "weighted_f1": float(
                finetuned_on_target_thresholded.weighted_f1
                - finetuned_on_target.weighted_f1
            ),
            "gmean": float(
                finetuned_on_target_thresholded.gmean - finetuned_on_target.gmean
            ),
        },
        "selected_mode": selected_mode,
        "selection_stability_score": (
            float(val_metrics.get("selection_stability", {}).get("selection_stability_score", 0.0))
            if isinstance(val_metrics.get("selection_stability"), dict)
            else 0.0
        ),
        "threshold_constraint_flags": val_metrics.get("threshold_constraint_flags"),
        "selection_reason": str(val_metrics.get("selection_reason")),
    }

    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"源域模型 → 源域测试集  Macro-F1: {source_on_source.macro_f1:.4f}")
    print(f"源域模型 → 目标域测试集 Macro-F1: {source_on_target.macro_f1:.4f}")
    print(f"微调模型 → 目标域测试集 Macro-F1: {finetuned_on_target.macro_f1:.4f}")
    print(f"微调改进: +{f1_improvement:.4f} ({f1_improvement * 100:.1f}%)")
    print(
        f"微调模型(阈值后) → 目标域测试集 Macro-F1: "
        f"{finetuned_on_target_thresholded.macro_f1:.4f}"
    )
    print(
        f"阈值后改进: +{float(f1_improvement_thresholded):.4f} "
        f"({float(f1_improvement_thresholded) * 100:.1f}%)"
    )

    # === 可视化产物 ===
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    class_names = _build_class_names_from_config()

    # 1. 双域混淆矩阵
    cm_source = np.array(source_on_source.confusion_matrix, dtype=np.int64)
    cm_target = np.array(finetuned_on_target.confusion_matrix, dtype=np.int64)

    dual_cm_path = eval_dir / "cm_source_vs_target_dual.png"
    saved_dual_cm = save_dual_confusion_matrix(
        cm_source=cm_source,
        cm_target=cm_target,
        output_path=dual_cm_path,
        class_names=class_names,
        normalize=True,
        source_name="Source",
        target_name="Target",
    )
    if saved_dual_cm is not None:
        print(f"  双域混淆矩阵已保存: {saved_dual_cm}")

    # 2. t-SNE 特征可视化
    print("\n[可视化] 提取特征用于 t-SNE...")
    try:
        source_features, source_labels = _extract_domain_features(
            run_dir, target_ckpt, domain="source", max_samples=4000
        )
        target_features, target_labels = _extract_domain_features(
            run_dir, target_ckpt, domain="target", max_samples=4000
        )

        if source_features.size > 0 and target_features.size > 0:
            tsne_path = eval_dir / "tsne_finetuned_source_target.png"
            saved_tsne = save_domain_class_tsne(
                source_features=source_features,
                source_labels=source_labels,
                target_features=target_features,
                target_labels=target_labels,
                class_names=class_names,
                output_path=tsne_path,
                max_samples_per_class=200,
                perplexity=30.0,
                source_name="Source",
                target_name="Target",
            )
            if saved_tsne is not None:
                print(f"  t-SNE 可视化已保存: {saved_tsne}")
    except Exception as exc:
        print(f"  [警告] t-SNE 可视化失败: {exc}")

    leaderboard_path = _write_target_test_leaderboard(
        eval_dir=eval_dir,
        raw_result=finetuned_on_target,
        thresholded_result=finetuned_on_target_thresholded,
        objective_text=objective_payload_text,
        selection_stability_score=(
            float(val_metrics.get("selection_stability", {}).get("selection_stability_score", 0.0))
            if isinstance(val_metrics.get("selection_stability"), dict)
            else None
        ),
        threshold_constraint_flags=(
            dict(val_metrics.get("threshold_constraint_flags", {}))
            if isinstance(val_metrics.get("threshold_constraint_flags"), dict)
            else None
        ),
    )
    print(f"  目标域测试榜单已保存: {leaderboard_path}")

    summary = {
        "source_on_source": source_on_source.__dict__,
        "source_on_target": source_on_target.__dict__,
        "finetuned_on_target": finetuned_on_target.__dict__,
        "f1_improvement": f1_improvement,
        "threshold_search": threshold_search_result,
        "finetuned_on_target_thresholded": finetuned_on_target_thresholded.__dict__,
        "f1_improvement_thresholded": f1_improvement_thresholded,
    }
    summary_path = eval_dir / "transfer_gap_summary.json"
    write_json(summary_path, summary)
    print(f"\n[评估] 对比摘要已保存: {summary_path}")
    return summary
