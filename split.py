from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from essence_forge.core.rflymad_index import (
    build_index as standalone_build_index,
    parse_filename as standalone_parse_filename,
    save_splits as standalone_save_splits,
    split_records as standalone_split_records,
)


def parse_filename(filename: str) -> Optional[Dict]:
    return standalone_parse_filename(filename)


def build_index(
    data_dir: Path,
    use_A: Tuple[int, ...],
    use_B: Tuple[int, ...],
    use_faults: Tuple[int, ...],
) -> List[Dict]:
    return standalone_build_index(
        data_dir=data_dir,
        use_A=use_A,
        use_B=use_B,
        use_faults=use_faults,
    )


def split_records(
    records: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_by_fault: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    return standalone_split_records(
        records=records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify_by_fault=stratify_by_fault,
    )


def save_splits(
    run_dir: Path,
    source_train: List[Dict],
    source_val: List[Dict],
    source_test: List[Dict],
    target_train: List[Dict],
    target_val: List[Dict],
    target_test: List[Dict],
) -> None:
    standalone_save_splits(
        run_dir=run_dir,
        source_train=source_train,
        source_val=source_val,
        source_test=source_test,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
    )
