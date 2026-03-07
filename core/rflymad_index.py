"""
rflymad_index.py

RflyMAD 数据集索引与切分模块

职责:
1. 扫描数据目录，解析文件名获取元信息
2. 根据源域/目标域配置过滤文件
3. 实现 train/val/test 分层切分
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split

import sys

from essence_forge.core.runtime_config import CFG
from essence_forge.core.utils import write_json


# =============================================================================
# 文件名解析
# =============================================================================

# 文件名格式: Case_[A][B][CD][EFGHIJ].csv
# 例如: Case_2006000001.csv
#   - A=2: HIL
#   - B=0: hover
#   - CD=06: gyroscope 故障
#   - EFGHIJ=000001: 序列号
FILENAME_PATTERN = re.compile(r"Case_(\d)(\d)(\d{2})(\d{6})\.csv")


def parse_filename(filename: str) -> Optional[Dict]:
    """
    解析 RflyMAD 文件名，提取元信息
    
    参数:
        filename: 文件名（不含路径）
        
    返回:
        dict 包含 A, B, fault_code, sequence
        如果解析失败返回 None
    
    示例:
        >>> parse_filename("Case_2006000001.csv")
        {'A': 2, 'B': 0, 'fault_code': 6, 'sequence': '000001'}
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    
    return {
        "A": int(match.group(1)),           # 子数据集：1=SIL, 2=HIL, 3=Real
        "B": int(match.group(2)),           # 飞行状态：0=hover, 1=waypoint, ...
        "fault_code": int(match.group(3)),  # 故障类型：00-10
        "sequence": match.group(4),         # 序列号
    }


# =============================================================================
# 数据集索引构建
# =============================================================================

def build_index(
    data_dir: Path,
    use_A: Tuple[int, ...],
    use_B: Tuple[int, ...],
    use_faults: Tuple[int, ...],
) -> List[Dict]:
    """
    扫描数据目录并构建索引
    
    参数:
        data_dir: 数据根目录
        use_A: 要使用的子数据集（如 (2,) 表示只用 HIL）
        use_B: 要使用的飞行状态（如 (0,) 表示只用 hover）
        use_faults: 要使用的故障类型（如 (0, 1, ..., 10)）
        
    返回:
        records 列表，每个 record 包含:
            - file_path: 文件绝对路径
            - filename: 文件名
            - A, B, fault_code, sequence: 解析的元信息
            - class_id: 映射后的类别 ID
    """
    records = []
    
    # 递归扫描所有 CSV 文件
    for csv_file in data_dir.rglob("Case_*.csv"):
        parsed = parse_filename(csv_file.name)
        if parsed is None:
            continue
        
        # 过滤
        if parsed["A"] not in use_A:
            continue
        if parsed["B"] not in use_B:
            continue
        if parsed["fault_code"] not in use_faults:
            continue
        
        # 映射故障代码到类别 ID
        class_id = CFG.fault_to_class.get(parsed["fault_code"])
        if class_id is None:
            continue
        
        records.append({
            "file_path": str(csv_file.resolve()),
            "filename": csv_file.name,
            "A": parsed["A"],
            "B": parsed["B"],
            "fault_code": parsed["fault_code"],
            "sequence": parsed["sequence"],
            "class_id": class_id,
        })
    
    return records


def split_records(
    records: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_by_fault: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    将记录切分为 train/val/test
    
    参数:
        records: 完整记录列表
        train_ratio, val_ratio, test_ratio: 切分比例
        seed: 随机种子
        stratify_by_fault: 是否按故障类别分层采样
        
    返回:
        (train_records, val_records, test_records)
    """
    if len(records) == 0:
        return [], [], []
    
    # 提取用于分层的标签
    labels = [r["class_id"] for r in records] if stratify_by_fault else None
    
    # 先分出 test
    test_size = test_ratio
    train_val_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    
    # 再从 train_val 中分出 val
    # 注意：val_ratio 是相对于原始数据的比例
    # 所以在 train_val 中的比例是 val_ratio / (train_ratio + val_ratio)
    val_size_in_train_val = val_ratio / (train_ratio + val_ratio)
    labels_train_val = [r["class_id"] for r in train_val_records] if stratify_by_fault else None
    
    train_records, val_records = train_test_split(
        train_val_records,
        test_size=val_size_in_train_val,
        random_state=seed,
        stratify=labels_train_val,
    )
    
    return train_records, val_records, test_records


def save_splits(
    run_dir: Path,
    source_train: List[Dict],
    source_val: List[Dict],
    source_test: List[Dict],
    target_train: List[Dict],
    target_val: List[Dict],
    target_test: List[Dict],
) -> None:
    """
    保存数据切分结果到 JSON 文件
    
    保存的文件:
        - split_source_train.json
        - split_source_val.json
        - split_source_test.json
        - split_target_train.json
        - split_target_val.json
        - split_target_test.json
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    
    write_json(run_dir / "split_source_train.json", source_train)
    write_json(run_dir / "split_source_val.json", source_val)
    write_json(run_dir / "split_source_test.json", source_test)
    write_json(run_dir / "split_target_train.json", target_train)
    write_json(run_dir / "split_target_val.json", target_val)
    write_json(run_dir / "split_target_test.json", target_test)
    
    # 打印统计信息
    print(f"[Source Domain] A={CFG.source_A}, B={CFG.source_B}")
    print(f"  Train: {len(source_train)}, Val: {len(source_val)}, Test: {len(source_test)}")
    _print_class_distribution(source_train, "  Train")
    
    print(f"[Target Domain] A={CFG.target_A}, B={CFG.target_B}")
    print(f"  Train: {len(target_train)}, Val: {len(target_val)}, Test: {len(target_test)}")
    _print_class_distribution(target_train, "  Train")


def _print_class_distribution(records: List[Dict], prefix: str = "") -> None:
    """打印类别分布"""
    counts = {}
    for r in records:
        class_id = r["class_id"]
        counts[class_id] = counts.get(class_id, 0) + 1
    
    dist_str = ", ".join(
        f"{CFG.class_id_to_name.get(k, str(k))}:{v}"
        for k, v in sorted(counts.items())
    )
    print(f"{prefix} 类别分布: {dist_str}")
