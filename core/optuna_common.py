"""
optuna_common.py

与 Optuna 调优相关、且不依赖 torch/sklearn 的公共工具。
"""

from __future__ import annotations

import re
from typing import Dict, List


def canonicalize_class_name(name: str) -> str:
    base = str(name).split("(", 1)[0].strip().lower()
    base = re.sub(r"[^a-z0-9_]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "class"


def build_expected_cost_keys(
    class_id_to_name: Dict[int, str],
    num_classes: int,
) -> List[str]:
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
