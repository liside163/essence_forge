from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from essence_forge.config import ExperimentConfig
from essence_forge.core.cross_sensor_residuals import (
    default_cross_sensor_residual_channel_names,
    fit_cross_sensor_residual_calibration,
    is_calibrated_cross_sensor_residual_scheme,
    normalize_cross_sensor_residual_scheme,
    write_cross_sensor_residual_fit,
)
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
CROSS_SENSOR_RESIDUAL_FIT_FILENAME = "cross_sensor_residual_fit.json"
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


def _remove_directory_tree(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        return
    except OSError as exc:
        fallback = path.with_name(f"{path.name}.stale.{uuid.uuid4().hex[:8]}")
        path.rename(fallback)
        try:
            shutil.rmtree(fallback)
        except OSError as cleanup_exc:
            print(
                f"[WARN] failed to remove {path} directly ({exc}); "
                f"renamed to {fallback} but cleanup also failed ({cleanup_exc})."
            )


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

    scheme = normalize_cross_sensor_residual_scheme(residual_cfg.get("scheme"))
    payload["scheme"] = scheme
    payload["clip_value"] = residual_cfg.get("clip_value", 6.0)
    payload["calibration_split"] = residual_cfg.get("calibration_split", "source_train_nofault")
    payload["max_lag_steps"] = residual_cfg.get("max_lag_steps", 4)
    window_norm = bool(residual_cfg.get("window_norm", True))
    if scheme == "legacy9":
        payload["window_norm"] = window_norm
        norm_eps = residual_cfg.get("norm_eps", 1e-6)
        if window_norm:
            try:
                payload["norm_eps"] = float(norm_eps)
            except (TypeError, ValueError):
                payload["norm_eps"] = norm_eps

    if "mask_mode" in residual_cfg:
        payload["mask_mode"] = residual_cfg.get("mask_mode")
    payload["channels"] = [
        {"name": name}
        for name in default_cross_sensor_residual_channel_names(scheme)
    ]
    payload["num_channels"] = len(default_cross_sensor_residual_channel_names(scheme))
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


def required_preprocess_artifact_files(runtime_payload: Dict[str, Any]) -> tuple[str, ...]:
    residual_cfg = runtime_payload.get("cross_sensor_residuals", {})
    if not isinstance(residual_cfg, dict):
        return PREPROCESS_ARTIFACT_FILES
    if not bool(residual_cfg.get("enable", False)):
        return PREPROCESS_ARTIFACT_FILES
    scheme = normalize_cross_sensor_residual_scheme(residual_cfg.get("scheme"))
    if not is_calibrated_cross_sensor_residual_scheme(scheme):
        return PREPROCESS_ARTIFACT_FILES
    return PREPROCESS_ARTIFACT_FILES + (CROSS_SENSOR_RESIDUAL_FIT_FILENAME,)


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
    scheme = normalize_cross_sensor_residual_scheme(
        getattr(CFG, "cross_sensor_residual_scheme", "legacy9")
    )
    residual_names = default_cross_sensor_residual_channel_names(scheme)
    CFG.values["cross_sensor_residual_scheme"] = scheme
    CFG.values["cross_sensor_residual_channels"] = len(residual_names)
    CFG.values["cross_sensor_residual_channel_names"] = residual_names
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


def _cross_sensor_residual_fit_required() -> bool:
    return bool(getattr(CFG, "use_cross_sensor_residuals", False)) and is_calibrated_cross_sensor_residual_scheme(
        getattr(CFG, "cross_sensor_residual_scheme", "legacy9")
    )


def _precomputed_ready(run_dir: Path) -> bool:
    return all((run_dir / "precomputed" / split / "manifest.json").exists() for split in PRECOMPUTED_SPLITS)


def _preprocess_artifacts_ready(run_dir: Path, runtime_payload: Dict[str, Any]) -> bool:
    return all((run_dir / name).exists() for name in required_preprocess_artifact_files(runtime_payload)) and _precomputed_ready(run_dir)


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


def _ensure_cross_sensor_residual_fit(run_dir: Path, force_rebuild: bool = False) -> Path | None:
    if not _cross_sensor_residual_fit_required():
        CFG.values["cross_sensor_residual_fit_path"] = ""
        return None

    fit_path = run_dir / CROSS_SENSOR_RESIDUAL_FIT_FILENAME
    if fit_path.exists() and not force_rebuild:
        CFG.values["cross_sensor_residual_fit_path"] = str(fit_path.resolve())
        return fit_path

    source_train = read_json(run_dir / "split_source_train.json")
    normal_class_id = int(getattr(CFG, "fault_to_class", {}).get(10, 10))
    normal_records = [
        record
        for record in source_train
        if int(record.get("class_id", -1)) == normal_class_id
    ]
    if len(normal_records) == 0:
        raise ValueError("source_train split does not contain no-fault records for residual calibration")

    loader = MissionLoader(max_cache_items=64)
    raw_sequences = [loader.load(record["file_path"]) for record in normal_records]
    fit_payload = fit_cross_sensor_residual_calibration(
        raw_sequences=raw_sequences,
        sample_rate_hz=float(getattr(CFG, "windows_sample_rate_hz", 120.0)),
        max_lag_steps=int(getattr(CFG, "cross_sensor_residual_max_lag_steps", 4)),
        channel_names=tuple(str(name) for name in getattr(CFG, "channel_names", ())),
    )
    write_cross_sensor_residual_fit(fit_path, fit_payload)
    CFG.values["cross_sensor_residual_fit_path"] = str(fit_path.resolve())
    return fit_path


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
        _remove_directory_tree(run_dir / "precomputed")

    _ensure_source_stats(run_dir)
    _ensure_cross_sensor_residual_fit(run_dir, force_rebuild=force_rebuild)
    if force_rebuild or not _preprocess_artifacts_ready(run_dir, runtime_payload):
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
