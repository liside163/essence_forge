"""
cost_sensitive.py

类别不平衡成本敏感训练工具模块。

职责:
1. 统一类别名 -> `cost_*` 键的规范映射，避免类别顺序错位。
2. 从 `optuna_cost_result.json` 读取最优成本与超参数。
3. 将 `best_costs` 安全转换为训练可用的成本张量。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch


def canonicalize_class_name(name: str) -> str:
    """
    将类别名规范化为可稳定用于 Optuna 参数键的字符串。

    为什么要做:
    1. 配置中的类别名可能含括号、空格、大小写差异，例如 `Motor(00)`、`GPS`。
    2. 如果不规范化，`best_costs` 与训练读取时的键名容易不一致，导致成本权重错位。

    示例:
    - `Motor(00)` -> `motor`
    - `GPS` -> `gps`
    - `no fault` -> `no_fault`
    """
    base = str(name).split("(", 1)[0].strip().lower()
    base = re.sub(r"[^a-z0-9_]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "class"


def build_expected_cost_keys(
    class_id_to_name: Dict[int, str],
    num_classes: int,
) -> List[str]:
    """
    基于 class_id 顺序构建期望的成本键列表。

    关键约束:
    1. 必须严格按 `class_id` 从小到大构建。
    2. 键名统一为 `cost_<规范化类别名>`。
    3. 规范化后若重名，自动追加后缀避免冲突。
    """
    normalized_names: List[str] = []
    seen: Dict[str, int] = {}

    for class_id in range(num_classes):
        raw_name = class_id_to_name.get(class_id, class_id_to_name.get(str(class_id), f"class_{class_id}"))
        normalized = canonicalize_class_name(raw_name)

        if normalized not in seen:
            seen[normalized] = 1
            normalized_names.append(normalized)
        else:
            seen[normalized] += 1
            normalized_names.append(f"{normalized}_{seen[normalized]}")

    return [f"cost_{name}" for name in normalized_names]


def load_optuna_cost_result(
    run_dir: Path,
    result_filename: str = "optuna_cost_result.json",
) -> Dict[str, Any]:
    """
    从运行目录读取 Optuna 成本优化结果。

    返回字段:
    - `best_costs`: dict，类别成本权重
    - `best_hparams`: dict，最优超参数（可选）
    """
    path = Path(run_dir) / result_filename
    if not path.exists():
        raise FileNotFoundError(f"未找到 Optuna 成本结果文件: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Optuna 成本结果格式错误，期望 dict，实际: {type(payload)!r}")

    best_costs = payload.get("best_costs")
    if not isinstance(best_costs, dict):
        raise KeyError("Optuna 成本结果缺少 `best_costs` 或类型不是 dict")

    best_hparams = payload.get("best_hparams")
    if best_hparams is None:
        payload["best_hparams"] = {}
    elif not isinstance(best_hparams, dict):
        raise TypeError("Optuna 成本结果字段 `best_hparams` 必须是 dict")

    return payload


def build_cost_tensor_from_best_costs(
    best_costs: Dict[str, Any],
    class_id_to_name: Dict[int, str],
    num_classes: int,
) -> torch.Tensor:
    """
    将 Optuna `best_costs` 转换为训练可用的成本张量。

    输出维度:
    - `costs`: [num_classes]

    为什么强调维度:
    - 训练时损失函数会按标签索引 `costs[targets]`，
      其中 `targets` 维度为 [B]，因此 `costs` 必须与类别轴一一对应。
    """
    expected_keys = build_expected_cost_keys(
        class_id_to_name=class_id_to_name,
        num_classes=num_classes,
    )
    missing_keys = [key for key in expected_keys if key not in best_costs]
    if missing_keys:
        raise KeyError(
            "Optuna best_costs 缺少关键成本项，无法保证类别对齐: "
            f"{missing_keys}"
        )

    ordered_costs = [float(best_costs[key]) for key in expected_keys]
    return torch.tensor(ordered_costs, dtype=torch.float32)


def extract_supported_hparams(best_hparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    过滤并返回主项目训练链路支持的超参数。

    兼容规则:
    - `dropout` (旧脚本字段) 会映射为 `tcn_dropout`。
    - 不支持字段（如 `classifier_dropout`）会自动忽略。
    """
    supported: Dict[str, Any] = {}
    if "learning_rate" in best_hparams:
        supported["learning_rate"] = float(best_hparams["learning_rate"])
    if "tcn_kernel_size" in best_hparams:
        supported["tcn_kernel_size"] = int(best_hparams["tcn_kernel_size"])
    if "tcn_channels" in best_hparams:
        supported["tcn_channels"] = int(best_hparams["tcn_channels"])

    if "tcn_dropout" in best_hparams:
        supported["tcn_dropout"] = float(best_hparams["tcn_dropout"])
    elif "dropout" in best_hparams:
        # 兼容旧版调优脚本字段命名
        supported["tcn_dropout"] = float(best_hparams["dropout"])

    return supported

