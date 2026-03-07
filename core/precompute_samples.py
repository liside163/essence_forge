"""
precompute_samples.py

离线预处理脚本 —— 将确定性数据处理一次性缓存为 .npy 文件。

用法:
    python tools/precompute_samples.py --run-dir <run_dir>

功能:
    1. 读取 split_source_train.json / split_source_val.json / source_stats.json
    2. 用 augment_mode="off" 构建 SourceDomainDataset（仅执行确定性处理）
    3. 遍历所有样本，序列化为 .npy + manifest.json
    4. 训练时由 PrecomputedDataset 直接读取
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from essence_forge.core.runtime_config import CFG
from essence_forge.core.rflymad_io import MissionLoader
from essence_forge.core.datasets import SourceDomainDataset, DeterministicWindowDataset, Sample
from essence_forge.core.utils import read_json


def _default_num_workers() -> int:
    cpu_count = int(os.cpu_count() or 1)
    return max(1, min(8, cpu_count))


def _collate_single_sample(batch: List[Sample]) -> Sample:
    if len(batch) != 1:
        raise ValueError(f"预计算 DataLoader 期望 batch_size=1，实际={len(batch)}")
    return batch[0]


def _precompute_split(
    split_name: str,
    records: List[Dict],
    mean: np.ndarray,
    std: np.ndarray,
    loader: MissionLoader,
    output_dir: Path,
    is_train: bool,
    base_seed: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> None:
    """
    预计算单个 split 的所有样本并保存到 output_dir。

    输出:
    - output_dir/sample_{idx:06d}.npy
    - output_dir/manifest.json
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用 augment_mode="off" 的 Dataset，仅执行确定性处理
    if is_train:
        dataset = SourceDomainDataset(
            records=records,
            zscore_mean=mean,
            zscore_std=std,
            loader=loader,
            is_train=False,  # 关键: 关闭所有增强
            base_seed=base_seed,
            enable_targeted_gan=False,
            enable_physics_augment=False,
            augment_mode="off",
        )
    else:
        dataset = DeterministicWindowDataset(
            records=records,
            zscore_mean=mean,
            zscore_std=std,
            loader=loader,
            windows_per_scale=int(
                getattr(CFG, "eval_windows_per_scale",
                        getattr(CFG, "train_windows_per_mission_per_scale", 2))
            ),
            base_seed=base_seed,
        )

    total = len(dataset)
    print(f"  [{split_name}] 共 {total} 个样本")

    worker_count = max(0, int(num_workers))
    loader_kwargs: Dict[str, object] = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": worker_count,
        "collate_fn": _collate_single_sample,
        "pin_memory": False,
    }
    if worker_count > 0:
        if int(prefetch_factor) > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    dataloader = DataLoader(**loader_kwargs)

    # 收集增强 context 所需的元信息
    channel_names = [ch.name for ch in CFG.channels]
    raw_channels = int(getattr(CFG, "input_dim", 19))

    manifest_samples: List[Dict] = []
    t0 = time.perf_counter()

    for idx, sample in enumerate(dataloader):
        x_np = sample.x.numpy().astype(np.float32)

        # 保存 .npy
        npy_name = f"sample_{idx:06d}.npy"
        npy_path = output_dir / npy_name
        np.save(npy_path, x_np)

        # 收集元信息（用于在线增强重建 AugContext）
        entry: Dict = {
            # Store relative path/file name to keep manifest cross-platform.
            "path": npy_name,
            "class_id": int(sample.y.item()),
            "length": int(sample.length),
            "window_length": int(x_np.shape[0]),
        }

        # 从 dataset 提取增强上下文信息
        if is_train and hasattr(dataset, "stage_records") and hasattr(dataset, "use_stage_windowing") and dataset.use_stage_windowing:
            spec = dataset.stage_records[idx]
            record = dataset.records[spec.record_idx]
            entry["fault_code"] = int(record.get("fault_code", 10))
            meta = loader.load_metadata(record["file_path"])
            onset_idx = int(meta.fault_onset_idx)
            if entry["fault_code"] != 10 and onset_idx < 0:
                onset_idx = int(round(
                    getattr(CFG, "windows_normal_context_seconds", 5.0)
                    * getattr(CFG, "windows_sample_rate_hz", 120)
                ))
            entry["fault_onset_idx"] = onset_idx
            entry["window_start"] = spec.start
        elif is_train and hasattr(dataset, "records"):
            # 传统模式
            samples_per_mission = getattr(dataset, "samples_per_mission", 1)
            mission_idx = idx // samples_per_mission
            if mission_idx < len(dataset.records):
                record = dataset.records[mission_idx]
                entry["fault_code"] = int(record.get("fault_code", 10))
                meta = loader.load_metadata(record["file_path"])
                onset_idx = int(meta.fault_onset_idx)
                if entry["fault_code"] != 10 and onset_idx < 0:
                    onset_idx = int(round(
                        getattr(CFG, "windows_normal_context_seconds", 5.0)
                        * getattr(CFG, "windows_sample_rate_hz", 120)
                    ))
                entry["fault_onset_idx"] = onset_idx
                entry["window_start"] = 0
        else:
            entry["fault_code"] = 10
            entry["fault_onset_idx"] = -1
            entry["window_start"] = 0

        manifest_samples.append(entry)

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            print(f"    [{split_name}] {idx + 1}/{total} ({rate:.1f} samples/s)")

    # 写 manifest
    manifest = {
        "version": 1,
        "split": split_name,
        "total_samples": total,
        "raw_channels": raw_channels,
        "channel_names": channel_names,
        "samples": manifest_samples,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"  [{split_name}] 完成, 耗时 {elapsed:.1f}s, manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="预计算训练样本缓存")
    parser.add_argument("--run-dir", type=str, required=True, help="实验运行目录")
    parser.add_argument("--base-seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_default_num_workers(),
        help="预计算并行 worker 数（WSL 推荐 8）",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="DataLoader 每个 worker 预取批次数（num_workers>0 时生效）",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 DataLoader persistent_workers（num_workers>0 时生效）",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="要预计算的 split (默认: train val)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir 不存在: {run_dir}")

    print(f"[precompute] run_dir: {run_dir}")
    print(f"[precompute] splits: {args.splits}")
    print(
        f"[precompute] workers={int(args.num_workers)} "
        f"prefetch_factor={int(args.prefetch_factor)} "
        f"persistent_workers={bool(args.persistent_workers)}"
    )

    # 加载统计量
    stats_path = run_dir / "source_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"缺少 source_stats.json: {stats_path}")
    stats = read_json(stats_path)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    loader = MissionLoader(max_cache_items=256)
    precomputed_root = run_dir / "precomputed"

    # 映射 split 名称到实际 JSON 文件
    # train/val -> split_source_train.json / split_source_val.json
    # target_train/target_val -> split_target_train.json / split_target_val.json
    def _resolve_split_file(split: str) -> Path:
        if split.startswith("target_"):
            return run_dir / f"split_{split}.json"
        return run_dir / f"split_source_{split}.json"

    for split in args.splits:
        split_file = _resolve_split_file(split)
        if not split_file.exists():
            print(f"[WARN] 跳过 {split}: {split_file} 不存在")
            continue

        records = read_json(split_file)
        print(f"\n[precompute] 处理 {split} ({len(records)} records)")

        _precompute_split(
            split_name=split,
            records=records,
            mean=mean,
            std=std,
            loader=loader,
            output_dir=precomputed_root / split,
            is_train=(split in {"train", "target_train"}),
            base_seed=args.base_seed,
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            persistent_workers=bool(args.persistent_workers),
        )

    print("\n[precompute] 全部完成!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
