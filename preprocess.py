from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from essence_forge.config import ExperimentConfig
from essence_forge.core.datasets import (
    _append_cross_sensor_residual_features as standalone_append_cross_sensor_residual_features,
    _normalize_window_with_stats as standalone_normalize_window_with_stats,
    build_stage_windows as standalone_build_stage_windows,
    generate_health_mask as standalone_generate_health_mask,
)
from essence_forge.core.precompute_samples import _precompute_split
from essence_forge.core.rflymad_io import MissionLoader, compute_zscore_stats
from essence_forge.core.runtime_config import CFG, reload_config
from essence_forge.core.utils import read_json, write_json


PRECOMPUTED_SPLITS = ("train", "val", "target_train", "target_val")
SPLIT_ARTIFACT_FILES = (
    "split_source_train.json",
    "split_source_val.json",
    "split_source_test.json",
    "split_target_train.json",
    "split_target_val.json",
    "split_target_test.json",
)
PREPROCESS_ARTIFACT_FILES = SPLIT_ARTIFACT_FILES + (
    "source_stats.json",
)


@dataclass(frozen=True)
class PreprocessArtifactsRef:
    shared_dir: Path
    fingerprint: str
    requested_link_mode: str
    effective_link_mode: str
    fingerprint_payload: Dict[str, Any]


def _normalize_cross_sensor_residuals_for_fingerprint(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    residual_cfg = (
        cfg.get("cross_sensor_residuals")
        if isinstance(cfg.get("cross_sensor_residuals"), dict)
        else {}
    )

    enable = bool(residual_cfg.get("enable", False))
    payload: Dict[str, Any] = {"enable": enable}
    if not enable:
        return payload

    window_norm = bool(residual_cfg.get("window_norm", True))
    payload["window_norm"] = window_norm
    if window_norm:
        norm_eps = residual_cfg.get("norm_eps", 1e-6)
        try:
            payload["norm_eps"] = float(norm_eps)
        except (TypeError, ValueError):
            payload["norm_eps"] = norm_eps

    for key in ("channels", "num_channels", "mask_mode"):
        if key in residual_cfg:
            payload[key] = residual_cfg.get(key)
    return payload


def _extract_preprocess_fingerprint_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    channels_cfg = cfg.get("channels", cfg.get("channels_19"))

    return {
        "paths": {
            "data_dir": paths_cfg.get("data_dir"),
        },
        "source_domain": cfg.get("source_domain"),
        "target_domain": cfg.get("target_domain"),
        "dataset": cfg.get("dataset"),
        "windows": cfg.get("windows"),
        "stats": cfg.get("stats"),
        "labels": {
            "fault_to_class": (
                cfg.get("labels", {}).get("fault_to_class")
                if isinstance(cfg.get("labels"), dict)
                else None
            ),
        },
        "channels": channels_cfg,
        "cross_sensor_residuals": _normalize_cross_sensor_residuals_for_fingerprint(cfg),
        "model": {
            "concat_health_mask_channels": model_cfg.get("concat_health_mask_channels"),
        },
    }


def compute_preprocess_fingerprint(cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    payload = _extract_preprocess_fingerprint_payload(cfg)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return fingerprint, payload


def build_stage_windows(
    total_length: int,
    fault_code: int,
    fault_onset_idx: int,
    window_len: int,
    stride: int,
    normal_class_id: int = 10,
    drop_prefault_normal_windows_for_fault_missions: bool = False,
):
    return standalone_build_stage_windows(
        total_length=total_length,
        fault_code=fault_code,
        fault_onset_idx=fault_onset_idx,
        window_len=window_len,
        stride=stride,
        normal_class_id=normal_class_id,
        drop_prefault_normal_windows_for_fault_missions=drop_prefault_normal_windows_for_fault_missions,
    )


def generate_health_mask(
    total_length: int,
    fault_code: int,
    fault_onset_idx: int,
    channel_names: Sequence[str],
) -> np.ndarray:
    return standalone_generate_health_mask(
        total_length=total_length,
        fault_code=fault_code,
        fault_onset_idx=fault_onset_idx,
        channel_names=channel_names,
    )


def normalize_window_with_stats(
    raw_window: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return standalone_normalize_window_with_stats(raw_window=raw_window, mean=mean, std=std)


def append_cross_sensor_residual_features(normalized_window: np.ndarray) -> np.ndarray:
    base_channels = int(normalized_window.shape[1])
    mean = np.zeros((base_channels,), dtype=np.float32)
    std = np.ones((base_channels,), dtype=np.float32)
    preserved = dict(CFG.values)
    CFG.values["input_dim"] = base_channels
    CFG.values["cross_sensor_residual_channels"] = 9
    CFG.values["windows_sample_rate_hz"] = 120
    CFG.values["cross_sensor_residual_window_norm"] = True
    CFG.values["cross_sensor_residual_norm_eps"] = 1e-6
    try:
        return standalone_append_cross_sensor_residual_features(
            normalized_window=normalized_window,
            mean=mean,
            std=std,
        )
    finally:
        CFG.values.clear()
        CFG.values.update(preserved)


def _resolve_split_path(run_dir: Path, split_name: str) -> Path:
    if split_name.startswith("target_"):
        return run_dir / f"split_{split_name}.json"
    return run_dir / f"split_source_{split_name}.json"


def _split_artifacts_ready(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in SPLIT_ARTIFACT_FILES)


def _precomputed_ready(run_dir: Path) -> bool:
    return all((run_dir / "precomputed" / split / "manifest.json").exists() for split in PRECOMPUTED_SPLITS)


def _write_preprocess_ref(
    run_dir: Path,
    fingerprint: str,
    fingerprint_payload: Dict[str, Any],
) -> None:
    write_json(
        run_dir / "preprocess_ref.json",
        {
            "shared_dir": str(run_dir.resolve()),
            "fingerprint": fingerprint,
            "requested_link_mode": "local",
            "effective_link_mode": "local",
            "fingerprint_payload": fingerprint_payload,
        },
    )


def _ensure_source_stats(run_dir: Path) -> None:
    if (run_dir / "source_stats.json").exists():
        return

    source_train = read_json(run_dir / "split_source_train.json")
    loader = MissionLoader(max_cache_items=32)
    mean, std = compute_zscore_stats(
        records=source_train,
        loader=loader,
        mode=CFG.stats_mode,
        max_missions=CFG.stats_max_missions,
        window_lengths=CFG.window_lengths,
        windows_per_mission_per_scale=CFG.stats_windows_per_mission_per_scale,
        seed=CFG.split_seed,
    )
    write_json(
        run_dir / "source_stats.json",
        {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    )


def _precompute_run_dir(
    run_dir: Path,
    precompute_num_workers: int,
    precompute_prefetch_factor: int,
    precompute_persistent_workers: bool,
) -> None:
    stats = read_json(run_dir / "source_stats.json")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    loader = MissionLoader(max_cache_items=256)
    precomputed_root = run_dir / "precomputed"
    precomputed_root.mkdir(parents=True, exist_ok=True)

    for split_name in PRECOMPUTED_SPLITS:
        split_path = _resolve_split_path(run_dir, split_name)
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file for precompute: {split_path}")
        records = read_json(split_path)
        _precompute_split(
            split_name=split_name,
            records=records,
            mean=mean,
            std=std,
            loader=loader,
            output_dir=precomputed_root / split_name,
            is_train=split_name in {"train", "target_train"},
            base_seed=CFG.run_seed,
            num_workers=int(max(0, precompute_num_workers)),
            prefetch_factor=int(max(1, precompute_prefetch_factor)),
            persistent_workers=bool(precompute_persistent_workers),
        )


def ensure_preprocess_artifacts(
    run_dir: Path,
    config: ExperimentConfig,
    force_rebuild: bool = False,
    link_mode: str = "local",
    precompute_num_workers: int = 0,
    precompute_prefetch_factor: int = 4,
    precompute_persistent_workers: bool = True,
) -> PreprocessArtifactsRef:
    snapshot_path = config.write_runtime_snapshot(run_dir)
    os.environ["ESSENCE_FORGE_CONFIG_PATH"] = str(snapshot_path)
    os.environ["UAV_TCN_CONFIG_PATH"] = str(snapshot_path)
    reload_config(snapshot_path)

    runtime_payload = config.runtime_payload()
    fingerprint, fingerprint_payload = compute_preprocess_fingerprint(runtime_payload)

    if not _split_artifacts_ready(run_dir):
        missing = [name for name in SPLIT_ARTIFACT_FILES if not (run_dir / name).exists()]
        raise FileNotFoundError(
            f"Split artifacts are missing under {run_dir}: {', '.join(missing)}"
        )

    if force_rebuild and (run_dir / "precomputed").exists():
        shutil.rmtree(run_dir / "precomputed")

    _ensure_source_stats(run_dir)
    if force_rebuild or not _precomputed_ready(run_dir):
        _precompute_run_dir(
            run_dir=run_dir,
            precompute_num_workers=precompute_num_workers,
            precompute_prefetch_factor=precompute_prefetch_factor,
            precompute_persistent_workers=precompute_persistent_workers,
        )

    _write_preprocess_ref(
        run_dir=run_dir,
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
    )
    return PreprocessArtifactsRef(
        shared_dir=run_dir.resolve(),
        fingerprint=fingerprint,
        requested_link_mode=link_mode,
        effective_link_mode="local",
        fingerprint_payload=fingerprint_payload,
    )
