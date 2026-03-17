from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from functools import cmp_to_key
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PACKAGE_DIR = _SCRIPT_DIR.parent
    _PROJECT_DIR = _PACKAGE_DIR.parent
    if str(_PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_DIR))
else:
    _PACKAGE_DIR = Path(__file__).resolve().parent.parent

if _PACKAGE_DIR.parent.parent.name == ".worktrees":
    _WORKSPACE_ROOT = _PACKAGE_DIR.parents[2]
else:
    _WORKSPACE_ROOT = _PACKAGE_DIR

from essence_forge.config import BUNDLED_CONFIG_PATH, ExperimentConfig, load_experiment_config
from essence_forge.preprocess import PRECOMPUTED_SPLITS, PREPROCESS_ARTIFACT_FILES, SPLIT_ARTIFACT_FILES, compute_preprocess_fingerprint
from essence_forge.run import cmd_eval, cmd_finetune, cmd_preprocess, cmd_train


BASELINE_EXPERIMENT_ID = "baseline"
SUMMARY_METRIC_KEY = "finetuned_model_thresholded_target.macro_f1"
SECONDARY_METRIC_KEY = "source_model_direct_target.macro_f1"
DEFAULT_CONFUSION_MATRIX_CELL = (0, 2)
DEFAULT_MANIFEST_PATH = Path("configs/ablations/core_structure.json")
DEFAULT_SUITE_DIR = Path("outputs/ablations/core_structure")
DEFAULT_EXISTING_BASELINE_RUN_DIR = _WORKSPACE_ROOT / "outputs" / "simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s"
PREPROCESS_REBUILD_EXPERIMENT_IDS = {
}
FORCE_BASELINE_PREPROCESS_EXPERIMENT_IDS = {
    "legacy_deform_e1",
    "raw_only_level0_e2",
    "grouped_shared_e3",
    "bounded_grouped_e4",
    "gated_mask_fusion_e5",
    "bounded_grouped_e4_ref",
    "e4_relax_reg",
    "e4_tight_reg",
    "e4_lr_fast",
    "e4_lr_slow",
    "e4_scale_wide",
    "e4_scale_tight",
    "e4_warmup_short",
    "e4_warmup_long",
    "e4_combo_soft",
    "e4_combo_soft_wide",
}


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def should_rebuild_preprocess(experiment_id: str) -> bool:
    return str(experiment_id) in PREPROCESS_REBUILD_EXPERIMENT_IDS


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_only(only: str | None) -> list[str] | None:
    if only is None or not only.strip():
        return None
    return [part.strip() for part in only.split(",") if part.strip()]


def _select_experiments(manifest: dict[str, Any], only: list[str] | None) -> list[dict[str, Any]]:
    experiments = list(manifest.get("experiments", []))
    if only is None:
        return experiments
    selected = [exp for exp in experiments if str(exp.get("id")) in set(only)]
    missing = [exp_id for exp_id in only if exp_id not in {str(exp.get("id")) for exp in selected}]
    if missing:
        raise ValueError(f"Unknown experiment ids: {missing}")
    return selected


def _build_experiment_config(
    *,
    base_config: ExperimentConfig,
    overrides: dict[str, Any],
    config_path: Path,
) -> ExperimentConfig:
    payload = _deep_merge(base_config.payload, overrides)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ExperimentConfig(
        name=base_config.name,
        title=base_config.title,
        config_path=config_path.resolve(),
        payload=payload,
        bundle_id=base_config.bundle_id,
        model_class_name=base_config.model_class_name,
    )


def _baseline_preprocess_source(suite_dir: Path) -> Path:
    suite_baseline = suite_dir / BASELINE_EXPERIMENT_ID
    if (suite_baseline / "precomputed").exists():
        return suite_baseline
    if DEFAULT_EXISTING_BASELINE_RUN_DIR.exists():
        return DEFAULT_EXISTING_BASELINE_RUN_DIR
    return DEFAULT_EXISTING_BASELINE_RUN_DIR


def _baseline_split_source(suite_dir: Path) -> Path:
    if _split_artifacts_ready(DEFAULT_EXISTING_BASELINE_RUN_DIR):
        return DEFAULT_EXISTING_BASELINE_RUN_DIR
    suite_baseline = suite_dir / BASELINE_EXPERIMENT_ID
    if _split_artifacts_ready(suite_baseline):
        return suite_baseline
    return DEFAULT_EXISTING_BASELINE_RUN_DIR


def _fingerprint_for_config(config: ExperimentConfig) -> str:
    fingerprint, _ = compute_preprocess_fingerprint(config.runtime_payload())
    return fingerprint


def _copy_file_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_split_artifacts(source_run_dir: Path, target_run_dir: Path) -> None:
    target_run_dir.mkdir(parents=True, exist_ok=True)
    for filename in SPLIT_ARTIFACT_FILES:
        _copy_file_if_exists(source_run_dir / filename, target_run_dir / filename)


def _remove_path_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.exists():
        shutil.rmtree(path)


def _link_or_copy_directory(source: Path, target: Path) -> str:
    _remove_path_if_exists(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_ref = os.path.relpath(source, start=target.parent)
        os.symlink(source_ref, target, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError):
        shutil.copytree(source, target)
        return "copy"


def _copy_preprocess_artifacts(source_run_dir: Path, target_run_dir: Path) -> None:
    target_run_dir.mkdir(parents=True, exist_ok=True)
    for filename in PREPROCESS_ARTIFACT_FILES:
        _copy_file_if_exists(source_run_dir / filename, target_run_dir / filename)
    preprocess_ref = source_run_dir / "preprocess_ref.json"
    _copy_file_if_exists(preprocess_ref, target_run_dir / "preprocess_ref.json")

    source_precomputed = source_run_dir / "precomputed"
    target_precomputed = target_run_dir / "precomputed"
    if source_precomputed.exists():
        mode = _link_or_copy_directory(source_precomputed, target_precomputed)
        print(
            f"[reuse-preprocess] {mode} precomputed: "
            f"{target_precomputed} -> {source_precomputed}"
        )


def _copy_full_run(source_run_dir: Path, target_run_dir: Path) -> None:
    if target_run_dir.exists():
        shutil.rmtree(target_run_dir)
    shutil.copytree(source_run_dir, target_run_dir)


def _copy_directory_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _load_existing_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "eval" / "essence_forge_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _nested_get(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _confusion_matrix_value(metrics: dict[str, Any], row_idx: int, col_idx: int) -> int | None:
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, list):
        return None
    if row_idx < 0 or row_idx >= len(matrix):
        return None
    row = matrix[row_idx]
    if not isinstance(row, list) or col_idx < 0 or col_idx >= len(row):
        return None
    value = row[col_idx]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _metric_or_neg_inf(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float("-inf")


def _cost_or_pos_inf(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return int(value)
    return 10**9


def _summary_ranking_config(manifest: dict[str, Any]) -> dict[str, Any]:
    ranking = manifest.get("ranking")
    if isinstance(ranking, dict):
        return ranking
    return {}


def _compare_rows(left: dict[str, Any], right: dict[str, Any], *, tie_epsilon: float) -> int:
    left_primary = _metric_or_neg_inf(left.get("primary_metric"))
    right_primary = _metric_or_neg_inf(right.get("primary_metric"))
    if abs(left_primary - right_primary) > tie_epsilon:
        return -1 if left_primary > right_primary else 1

    left_secondary = _metric_or_neg_inf(left.get("source_direct_macro_f1"))
    right_secondary = _metric_or_neg_inf(right.get("source_direct_macro_f1"))
    if left_secondary != right_secondary:
        return -1 if left_secondary > right_secondary else 1

    left_confusion = _cost_or_pos_inf(left.get("confusion_0_2"))
    right_confusion = _cost_or_pos_inf(right.get("confusion_0_2"))
    if left_confusion != right_confusion:
        return -1 if left_confusion < right_confusion else 1

    left_changes = _cost_or_pos_inf(left.get("param_change_count"))
    right_changes = _cost_or_pos_inf(right.get("param_change_count"))
    if left_changes != right_changes:
        return -1 if left_changes < right_changes else 1

    left_id = str(left.get("experiment_id", ""))
    right_id = str(right.get("experiment_id", ""))
    if left_id == right_id:
        return 0
    return -1 if left_id < right_id else 1


def _split_artifacts_ready(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in SPLIT_ARTIFACT_FILES)


def _preprocess_artifacts_ready(run_dir: Path) -> bool:
    if not all((run_dir / name).exists() for name in PREPROCESS_ARTIFACT_FILES):
        return False
    return all((run_dir / "precomputed" / split / "manifest.json").exists() for split in PRECOMPUTED_SPLITS)


def _run_training_pipeline(run_dir: Path, config: ExperimentConfig, *, force_rebuild: bool) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    source_checkpoint = checkpoints_dir / "tcn_source_essence_forge.pt"
    target_checkpoint = checkpoints_dir / "tcn_finetuned_essence_forge.pt"

    if not _split_artifacts_ready(run_dir):
        raise FileNotFoundError(
            "Missing split artifacts for ablation run. Reuse existing split files instead of rebuilding them."
        )
    if force_rebuild or not _preprocess_artifacts_ready(run_dir):
        cmd_preprocess(run_dir=run_dir, config=config, force_rebuild=force_rebuild)
    if force_rebuild or not source_checkpoint.exists():
        cmd_train(run_dir=run_dir, config=config)
    if force_rebuild or not target_checkpoint.exists():
        cmd_finetune(run_dir=run_dir, config=config)
    if force_rebuild or _load_existing_summary(run_dir) is None:
        cmd_eval(run_dir=run_dir, config=config)


def _seed_baseline_if_available(run_dir: Path, *, force_rebuild: bool) -> bool:
    if force_rebuild or run_dir.exists():
        return False
    if not DEFAULT_EXISTING_BASELINE_RUN_DIR.exists():
        return False
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in PREPROCESS_ARTIFACT_FILES:
        _copy_file_if_exists(
            DEFAULT_EXISTING_BASELINE_RUN_DIR / filename,
            run_dir / filename,
        )
    _copy_file_if_exists(
        DEFAULT_EXISTING_BASELINE_RUN_DIR / "preprocess_ref.json",
        run_dir / "preprocess_ref.json",
    )
    _copy_directory_if_exists(
        DEFAULT_EXISTING_BASELINE_RUN_DIR / "eval",
        run_dir / "eval",
    )
    _copy_directory_if_exists(
        DEFAULT_EXISTING_BASELINE_RUN_DIR / "logs",
        run_dir / "logs",
    )
    _copy_directory_if_exists(
        DEFAULT_EXISTING_BASELINE_RUN_DIR / "checkpoints",
        run_dir / "checkpoints",
    )
    _copy_directory_if_exists(
        DEFAULT_EXISTING_BASELINE_RUN_DIR / "configs",
        run_dir / "configs",
    )
    return True


def _ensure_baseline_preprocess_ready(
    *,
    base_config: ExperimentConfig,
    baseline_experiment: dict[str, Any],
    suite_dir: Path,
) -> None:
    run_dir = suite_dir / BASELINE_EXPERIMENT_ID
    config_path = run_dir / "configs" / "ablation_input.json"
    baseline_config = _build_experiment_config(
        base_config=base_config,
        overrides=dict(baseline_experiment.get("overrides", {})),
        config_path=config_path,
    )

    if not _split_artifacts_ready(run_dir):
        split_source = _baseline_split_source(suite_dir)
        if split_source.exists():
            _copy_split_artifacts(split_source, run_dir)

    if not _preprocess_artifacts_ready(run_dir):
        cmd_preprocess(run_dir=run_dir, config=baseline_config, force_rebuild=False)


def run_experiment(
    *,
    base_config: ExperimentConfig,
    experiment: dict[str, Any],
    suite_dir: Path,
    base_fingerprint: str,
    force_rebuild: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    experiment_id = str(experiment["id"])
    run_dir = suite_dir / experiment_id

    existing_summary = _load_existing_summary(run_dir)
    if skip_existing and existing_summary is not None:
        return {
            "experiment_id": experiment_id,
            "run_dir": str(run_dir.resolve()),
            "summary": existing_summary,
            "reused_preprocess": bool(experiment_id == BASELINE_EXPERIMENT_ID or not should_rebuild_preprocess(experiment_id)),
        }

    config_path = run_dir / "configs" / "ablation_input.json"
    experiment_config = _build_experiment_config(
        base_config=base_config,
        overrides=dict(experiment.get("overrides", {})),
        config_path=config_path,
    )
    fingerprint = _fingerprint_for_config(experiment_config)
    reused_preprocess = False

    if not _split_artifacts_ready(run_dir):
        split_source = _baseline_split_source(suite_dir)
        if split_source.exists():
            _copy_split_artifacts(split_source, run_dir)

    if (
        experiment_id != BASELINE_EXPERIMENT_ID
        and experiment_id in FORCE_BASELINE_PREPROCESS_EXPERIMENT_IDS
    ):
        preprocess_source = _baseline_preprocess_source(suite_dir)
        if not _preprocess_artifacts_ready(preprocess_source):
            raise FileNotFoundError(
                f"Missing E0 preprocess artifacts under {preprocess_source}. "
                "Run baseline preprocess first or launch the suite so E0 can be prepared."
            )
        _copy_preprocess_artifacts(preprocess_source, run_dir)
        reused_preprocess = True
    elif (
        experiment_id != BASELINE_EXPERIMENT_ID
        and not force_rebuild
        and not should_rebuild_preprocess(experiment_id)
        and fingerprint == base_fingerprint
    ):
        preprocess_source = _baseline_preprocess_source(suite_dir)
        if preprocess_source.exists():
            _copy_preprocess_artifacts(preprocess_source, run_dir)
            reused_preprocess = True

    _run_training_pipeline(
        run_dir=run_dir,
        config=experiment_config,
        force_rebuild=force_rebuild and not reused_preprocess,
    )
    summary = _load_existing_summary(run_dir)
    if summary is None:
        raise FileNotFoundError(f"Missing summary for experiment '{experiment_id}' under {run_dir}")
    return {
        "experiment_id": experiment_id,
        "run_dir": str(run_dir.resolve()),
        "summary": summary,
        "reused_preprocess": reused_preprocess,
    }


def _metrics_from_summary(summary: dict[str, Any], key: str) -> dict[str, Any]:
    value = summary.get(key)
    return value if isinstance(value, dict) else {}


def build_suite_summary(
    *,
    manifest: dict[str, Any],
    results: dict[str, dict[str, Any]],
    suite_dir: Path,
) -> dict[str, Any]:
    ranking_config = _summary_ranking_config(manifest)
    primary_metric_key = str(manifest.get("summary_metric_key", SUMMARY_METRIC_KEY))
    secondary_metric_key = str(ranking_config.get("secondary_metric_key", SECONDARY_METRIC_KEY))
    reference_experiment_id = str(ranking_config.get("reference_experiment_id", BASELINE_EXPERIMENT_ID))
    tie_epsilon = float(ranking_config.get("primary_metric_tie_epsilon", 0.0))
    confusion_matrix_cell = ranking_config.get("confusion_matrix_cell", list(DEFAULT_CONFUSION_MATRIX_CELL))
    if not isinstance(confusion_matrix_cell, (list, tuple)) or len(confusion_matrix_cell) != 2:
        confusion_matrix_cell = list(DEFAULT_CONFUSION_MATRIX_CELL)
    confusion_row = int(confusion_matrix_cell[0])
    confusion_col = int(confusion_matrix_cell[1])
    multi_seed_gate = ranking_config.get("multi_seed_gate", {})

    baseline_primary = None
    baseline_result = results.get(BASELINE_EXPERIMENT_ID)
    if baseline_result is not None:
        baseline_primary = _nested_get(baseline_result["summary"], primary_metric_key)

    reference_primary = None
    reference_confusion = None
    reference_result = results.get(reference_experiment_id)
    if reference_result is not None:
        reference_primary = _nested_get(reference_result["summary"], primary_metric_key)
        reference_thresholded = _metrics_from_summary(
            reference_result["summary"],
            "finetuned_model_thresholded_target",
        )
        reference_confusion = _confusion_matrix_value(
            reference_thresholded,
            confusion_row,
            confusion_col,
        )

    rows: list[dict[str, Any]] = []
    for experiment in manifest.get("experiments", []):
        experiment_id = str(experiment["id"])
        result = results.get(experiment_id)
        if result is None:
            continue
        summary = result["summary"]
        raw_metrics = _metrics_from_summary(summary, "finetuned_model_target")
        thresholded_metrics = _metrics_from_summary(summary, "finetuned_model_thresholded_target")
        source_direct_metrics = _metrics_from_summary(summary, "source_model_direct_target")
        if not thresholded_metrics:
            thresholded_metrics = raw_metrics
        primary_metric = _nested_get(summary, primary_metric_key)
        delta_vs_baseline = None
        if baseline_primary is not None and primary_metric is not None:
            delta_vs_baseline = float(primary_metric) - float(baseline_primary)
        delta_vs_reference = None
        if reference_primary is not None and primary_metric is not None:
            delta_vs_reference = float(primary_metric) - float(reference_primary)
        selection_meta = experiment.get("selection", {})
        if not isinstance(selection_meta, dict):
            selection_meta = {}
        source_direct_macro_f1 = _nested_get(summary, secondary_metric_key)
        confusion_0_2 = _confusion_matrix_value(
            thresholded_metrics,
            confusion_row,
            confusion_col,
        )
        eligible_for_multiseed = None
        if (
            isinstance(multi_seed_gate, dict)
            and experiment_id not in {BASELINE_EXPERIMENT_ID, reference_experiment_id}
        ):
            min_primary = multi_seed_gate.get("min_thresholded_macro_f1")
            min_secondary = multi_seed_gate.get("min_source_direct_macro_f1")
            eligible_for_multiseed = True
            if min_primary is not None:
                eligible_for_multiseed = (
                    eligible_for_multiseed
                    and primary_metric is not None
                    and float(primary_metric) >= float(min_primary)
                )
            if min_secondary is not None:
                eligible_for_multiseed = (
                    eligible_for_multiseed
                    and source_direct_macro_f1 is not None
                    and float(source_direct_macro_f1) >= float(min_secondary)
                )
            if reference_confusion is not None and confusion_0_2 is not None:
                eligible_for_multiseed = eligible_for_multiseed and confusion_0_2 <= reference_confusion
            elif reference_confusion is not None:
                eligible_for_multiseed = False
        rows.append(
            {
                "experiment_id": experiment_id,
                "description": str(experiment.get("description", "")),
                "round": selection_meta.get("round"),
                "param_change_count": selection_meta.get("param_change_count"),
                "run_dir": result["run_dir"],
                "primary_metric": primary_metric,
                "thresholded_macro_f1": thresholded_metrics.get("macro_f1"),
                "source_direct_macro_f1": source_direct_macro_f1,
                "confusion_0_2": confusion_0_2,
                "accuracy": thresholded_metrics.get("accuracy"),
                "gmean": thresholded_metrics.get("gmean"),
                "raw_macro_f1": raw_metrics.get("macro_f1"),
                "delta_vs_baseline": delta_vs_baseline,
                "delta_vs_reference": delta_vs_reference,
                "eligible_for_multiseed": eligible_for_multiseed,
                "reused_preprocess": bool(result.get("reused_preprocess", False)),
            }
        )

    rows.sort(key=cmp_to_key(lambda left, right: _compare_rows(left, right, tie_epsilon=tie_epsilon)))

    summary: dict[str, Any] = {
        "suite": manifest.get("suite", "core_structure"),
        "suite_dir": str(suite_dir.resolve()),
        "baseline_experiment_id": BASELINE_EXPERIMENT_ID,
        "reference_experiment_id": reference_experiment_id,
        "primary_metric": primary_metric_key,
        "secondary_metric": secondary_metric_key,
        "confusion_matrix_cell": [confusion_row, confusion_col],
        "primary_metric_tie_epsilon": tie_epsilon,
        "rows": rows,
    }
    if isinstance(multi_seed_gate, dict) and len(multi_seed_gate) > 0:
        summary["multi_seed_gate"] = {
            "min_thresholded_macro_f1": multi_seed_gate.get("min_thresholded_macro_f1"),
            "min_source_direct_macro_f1": multi_seed_gate.get("min_source_direct_macro_f1"),
            "reference_confusion_0_2": reference_confusion,
        }
    return summary


def _write_summary_json(summary: dict[str, Any], suite_dir: Path) -> Path:
    path = suite_dir / "suite_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_summary_csv(summary: dict[str, Any], suite_dir: Path) -> Path:
    path = suite_dir / "suite_summary.csv"
    rows = list(summary.get("rows", []))
    fieldnames = [
        "experiment_id",
        "description",
        "round",
        "param_change_count",
        "primary_metric",
        "thresholded_macro_f1",
        "source_direct_macro_f1",
        "confusion_0_2",
        "raw_macro_f1",
        "accuracy",
        "gmean",
        "delta_vs_baseline",
        "delta_vs_reference",
        "eligible_for_multiseed",
        "reused_preprocess",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def _write_summary_markdown(summary: dict[str, Any], suite_dir: Path) -> Path:
    path = suite_dir / "suite_summary.md"
    rows = list(summary.get("rows", []))
    lines = [
        "# Core Structure Ablation Summary",
        "",
        f"- Baseline: `{summary['baseline_experiment_id']}`",
        f"- Reference: `{summary.get('reference_experiment_id', summary['baseline_experiment_id'])}`",
        f"- Primary metric: `{summary['primary_metric']}`",
        f"- Secondary metric: `{summary.get('secondary_metric', SECONDARY_METRIC_KEY)}`",
        f"- Confusion cell: `{tuple(summary.get('confusion_matrix_cell', list(DEFAULT_CONFUSION_MATRIX_CELL)))}`",
        "",
        "| Experiment | Thresholded Macro-F1 | Direct-Transfer Macro-F1 | CM[0,2] | Raw Macro-F1 | Accuracy | G-Mean | Delta vs Reference | Eligible | Reused Preprocess |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {experiment_id} | {thresholded_macro_f1:.4f} | {source_direct_macro_f1:.4f} | {confusion_0_2} | {raw_macro_f1:.4f} | {accuracy:.4f} | {gmean:.4f} | {delta} | {eligible} | {reused_preprocess} |".format(
                experiment_id=row["experiment_id"],
                thresholded_macro_f1=float(row["thresholded_macro_f1"] or 0.0),
                source_direct_macro_f1=float(row["source_direct_macro_f1"] or 0.0),
                confusion_0_2=(
                    str(int(row["confusion_0_2"]))
                    if row["confusion_0_2"] is not None
                    else "N/A"
                ),
                raw_macro_f1=float(row["raw_macro_f1"] or 0.0),
                accuracy=float(row["accuracy"] or 0.0),
                gmean=float(row["gmean"] or 0.0),
                delta=(
                    f"{float(row['delta_vs_reference']):+.4f}"
                    if row["delta_vs_reference"] is not None
                    else "N/A"
                ),
                eligible=(
                    "yes"
                    if row["eligible_for_multiseed"] is True
                    else "no" if row["eligible_for_multiseed"] is False else "N/A"
                ),
                reused_preprocess="yes" if row["reused_preprocess"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_suite_summary_files(summary: dict[str, Any], suite_dir: Path) -> dict[str, str]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    return {
        "json": str(_write_summary_json(summary, suite_dir).resolve()),
        "csv": str(_write_summary_csv(summary, suite_dir).resolve()),
        "md": str(_write_summary_markdown(summary, suite_dir).resolve()),
    }


def run_suite(
    *,
    manifest_path: Path,
    suite_dir: Path,
    only: list[str] | None,
    force_rebuild: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    suite_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_experiment_config(manifest.get("base_config", str(BUNDLED_CONFIG_PATH)))
    base_fingerprint = _fingerprint_for_config(base_config)
    selected_experiments = _select_experiments(manifest, only)
    baseline_experiment = next(
        (exp for exp in manifest.get("experiments", []) if str(exp.get("id")) == BASELINE_EXPERIMENT_ID),
        None,
    )
    if baseline_experiment is None:
        raise ValueError("Manifest missing required baseline experiment")
    if any(
        str(exp.get("id")) in FORCE_BASELINE_PREPROCESS_EXPERIMENT_IDS
        for exp in selected_experiments
    ):
        _ensure_baseline_preprocess_ready(
            base_config=base_config,
            baseline_experiment=baseline_experiment,
            suite_dir=suite_dir,
        )
    results: dict[str, dict[str, Any]] = {}
    for experiment in selected_experiments:
        result = run_experiment(
            base_config=base_config,
            experiment=experiment,
            suite_dir=suite_dir,
            base_fingerprint=base_fingerprint,
            force_rebuild=force_rebuild,
            skip_existing=skip_existing,
        )
        results[str(experiment["id"])] = result

    summary = build_suite_summary(manifest=manifest, results=results, suite_dir=suite_dir)
    summary["artifacts"] = write_suite_summary_files(summary, suite_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Essence Forge core-structure ablation suite.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_suite(
        manifest_path=args.manifest,
        suite_dir=args.suite_dir,
        only=_parse_only(args.only),
        force_rebuild=bool(args.force_rebuild),
        skip_existing=bool(args.skip_existing),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
