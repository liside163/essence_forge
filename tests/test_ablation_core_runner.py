from __future__ import annotations

import json
import os
from pathlib import Path

from essence_forge.config import load_experiment_config
from scripts.core_structure_ablation import (
    BASELINE_EXPERIMENT_ID,
    SUMMARY_METRIC_KEY,
    _copy_preprocess_artifacts,
    _baseline_preprocess_source,
    build_suite_summary,
    load_manifest,
    run_experiment,
    should_rebuild_preprocess,
)


def test_manifest_contains_expected_core_structure_ablation_ids() -> None:
    manifest_path = Path("configs/ablations/core_structure.json")

    manifest = load_manifest(manifest_path)

    assert manifest["suite"] == "core_structure"
    assert [item["id"] for item in manifest["experiments"]] == [
        "baseline",
        "legacy_deform_e1",
        "raw_only_level0_e2",
        "grouped_shared_e3",
        "bounded_grouped_e4",
        "gated_mask_fusion_e5",
    ]


def test_manifest_contains_expected_e4_boost_ids() -> None:
    manifest_path = Path("configs/ablations/e4_boost.json")

    manifest = load_manifest(manifest_path)

    assert manifest["suite"] == "e4_boost"
    assert [item["id"] for item in manifest["experiments"]] == [
        "baseline",
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
    ]


def test_rebuild_preprocess_only_for_fingerprint_changing_experiments() -> None:
    assert should_rebuild_preprocess("baseline") is False
    assert should_rebuild_preprocess("legacy_deform_e1") is False
    assert should_rebuild_preprocess("raw_only_level0_e2") is False
    assert should_rebuild_preprocess("grouped_shared_e3") is False
    assert should_rebuild_preprocess("bounded_grouped_e4") is False
    assert should_rebuild_preprocess("gated_mask_fusion_e5") is False


def test_default_main_config_disables_cross_sensor_residuals() -> None:
    payload = json.loads(
        Path("configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["cross_sensor_residuals"]["enable"] is False


def test_build_suite_summary_uses_thresholded_macro_f1_and_baseline_delta(tmp_path) -> None:
    manifest = {
        "suite": "core_structure",
        "experiments": [
            {"id": "baseline", "description": "Baseline"},
            {"id": "no_freq_branch", "description": "No frequency branch"},
        ],
    }
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()

    results = {
        "baseline": {
            "run_dir": str(suite_dir / "baseline"),
            "summary": {
                "finetuned_model_target": {
                    "macro_f1": 0.861,
                    "accuracy": 0.913,
                    "gmean": 0.901,
                },
                "finetuned_model_thresholded_target": {
                    "macro_f1": 0.8743,
                    "accuracy": 0.9178,
                    "gmean": 0.9240,
                },
            },
            "reused_preprocess": True,
        },
        "no_freq_branch": {
            "run_dir": str(suite_dir / "no_freq_branch"),
            "summary": {
                "finetuned_model_target": {
                    "macro_f1": 0.811,
                    "accuracy": 0.884,
                    "gmean": 0.842,
                },
                "finetuned_model_thresholded_target": {
                    "macro_f1": 0.8301,
                    "accuracy": 0.891,
                    "gmean": 0.854,
                },
            },
            "reused_preprocess": True,
        },
    }

    summary = build_suite_summary(manifest=manifest, results=results, suite_dir=suite_dir)

    assert summary["baseline_experiment_id"] == BASELINE_EXPERIMENT_ID
    assert summary["primary_metric"] == SUMMARY_METRIC_KEY
    assert [row["experiment_id"] for row in summary["rows"]] == ["baseline", "no_freq_branch"]
    assert summary["rows"][0]["primary_metric"] == 0.8743
    assert summary["rows"][1]["delta_vs_baseline"] == 0.8301 - 0.8743
    assert summary["rows"][1]["reused_preprocess"] is True


def test_build_suite_summary_applies_e4_tie_break_and_gate(tmp_path) -> None:
    manifest = {
        "suite": "e4_boost",
        "summary_metric_key": "finetuned_model_thresholded_target.macro_f1",
        "ranking": {
            "reference_experiment_id": "bounded_grouped_e4_ref",
            "secondary_metric_key": "source_model_direct_target.macro_f1",
            "confusion_matrix_cell": [0, 2],
            "primary_metric_tie_epsilon": 0.0005,
            "multi_seed_gate": {
                "min_thresholded_macro_f1": 0.8771,
                "min_source_direct_macro_f1": 0.4417,
            },
        },
        "experiments": [
            {"id": "baseline", "description": "Baseline", "selection": {"round": "baseline", "param_change_count": 100}},
            {"id": "bounded_grouped_e4_ref", "description": "Reference", "selection": {"round": "round1", "param_change_count": 0}},
            {"id": "e4_lr_fast", "description": "Fast", "selection": {"round": "round1", "param_change_count": 1}},
            {"id": "e4_relax_reg", "description": "Relax", "selection": {"round": "round1", "param_change_count": 2}},
            {"id": "e4_combo_soft", "description": "Combo", "selection": {"round": "round2", "param_change_count": 4}},
        ],
    }
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()

    def _summary(thresholded_macro_f1, direct_macro_f1, confusion_0_2):
        return {
            "finetuned_model_target": {
                "macro_f1": thresholded_macro_f1 - 0.01,
                "accuracy": 0.88,
                "gmean": 0.85,
            },
            "finetuned_model_thresholded_target": {
                "macro_f1": thresholded_macro_f1,
                "accuracy": 0.89,
                "gmean": 0.86,
                "confusion_matrix": [
                    [10, 0, confusion_0_2],
                    [0, 8, 0],
                    [0, 0, 5],
                ],
            },
            "source_model_direct_target": {
                "macro_f1": direct_macro_f1,
            },
        }

    results = {
        "baseline": {
            "run_dir": str(suite_dir / "baseline"),
            "summary": _summary(0.8600, 0.4300, 90),
            "reused_preprocess": True,
        },
        "bounded_grouped_e4_ref": {
            "run_dir": str(suite_dir / "bounded_grouped_e4_ref"),
            "summary": _summary(0.8751, 0.4417, 79),
            "reused_preprocess": True,
        },
        "e4_lr_fast": {
            "run_dir": str(suite_dir / "e4_lr_fast"),
            "summary": _summary(0.8754, 0.4450, 82),
            "reused_preprocess": True,
        },
        "e4_relax_reg": {
            "run_dir": str(suite_dir / "e4_relax_reg"),
            "summary": _summary(0.8756, 0.4440, 78),
            "reused_preprocess": True,
        },
        "e4_combo_soft": {
            "run_dir": str(suite_dir / "e4_combo_soft"),
            "summary": _summary(0.8773, 0.4425, 79),
            "reused_preprocess": True,
        },
    }

    summary = build_suite_summary(manifest=manifest, results=results, suite_dir=suite_dir)

    assert summary["reference_experiment_id"] == "bounded_grouped_e4_ref"
    assert summary["confusion_matrix_cell"] == [0, 2]
    assert summary["multi_seed_gate"]["reference_confusion_0_2"] == 79
    assert [row["experiment_id"] for row in summary["rows"]] == [
        "e4_combo_soft",
        "e4_lr_fast",
        "e4_relax_reg",
        "bounded_grouped_e4_ref",
        "baseline",
    ]
    combo_row = summary["rows"][0]
    assert combo_row["eligible_for_multiseed"] is True
    assert combo_row["delta_vs_reference"] == 0.8773 - 0.8751
    assert summary["rows"][1]["eligible_for_multiseed"] is False
    assert summary["rows"][1]["source_direct_macro_f1"] == 0.4450
    assert summary["rows"][2]["confusion_0_2"] == 78


def test_baseline_preprocess_source_uses_workspace_baseline_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    import scripts.core_structure_ablation as ablation_module

    workspace_baseline = tmp_path / "workspace-baseline"
    workspace_baseline.mkdir()
    suite_baseline = tmp_path / "suite" / "baseline"
    suite_baseline_precomputed = suite_baseline / "precomputed"
    suite_baseline_precomputed.mkdir(parents=True)

    monkeypatch.setattr(
        ablation_module,
        "DEFAULT_EXISTING_BASELINE_RUN_DIR",
        workspace_baseline,
    )

    assert _baseline_preprocess_source(tmp_path / "suite") == suite_baseline


def test_run_experiment_reuses_existing_split_artifacts_before_pipeline(
    monkeypatch,
    tmp_path,
) -> None:
    import scripts.core_structure_ablation as ablation_module

    shared_run_dir = tmp_path / "shared"
    shared_run_dir.mkdir()
    for filename in ablation_module.SPLIT_ARTIFACT_FILES:
        (shared_run_dir / filename).write_text(json.dumps([{"stub": filename}]), encoding="utf-8")
    suite_baseline = tmp_path / "suite" / "baseline"
    suite_baseline.mkdir(parents=True)
    for filename in ablation_module.PREPROCESS_ARTIFACT_FILES:
        (suite_baseline / filename).write_text(json.dumps({"stub": filename}), encoding="utf-8")
    for split in ablation_module.PRECOMPUTED_SPLITS:
        split_dir = suite_baseline / "precomputed" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "manifest.json").write_text(json.dumps({"split": split}), encoding="utf-8")

    monkeypatch.setattr(ablation_module, "DEFAULT_EXISTING_BASELINE_RUN_DIR", shared_run_dir)

    def fake_run_training_pipeline(run_dir, config, *, force_rebuild):
        del config, force_rebuild
        for filename in ablation_module.SPLIT_ARTIFACT_FILES:
            assert (run_dir / filename).exists(), filename
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_path = eval_dir / "essence_forge_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "finetuned_model_target": {"macro_f1": 0.5, "accuracy": 0.5, "gmean": 0.5},
                    "finetuned_model_thresholded_target": {
                        "macro_f1": 0.6,
                        "accuracy": 0.6,
                        "gmean": 0.6,
                    },
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ablation_module, "_run_training_pipeline", fake_run_training_pipeline)

    result = run_experiment(
        base_config=load_experiment_config(),
        experiment={
            "id": "legacy_deform_e1",
            "description": "Legacy deformable baseline on strong input setting",
            "overrides": {"model": {"use_deformable_tcn": True}},
        },
        suite_dir=tmp_path / "suite",
        base_fingerprint="irrelevant",
        force_rebuild=False,
        skip_existing=False,
    )

    assert result["experiment_id"] == "legacy_deform_e1"


def test_run_experiment_forces_reuse_of_suite_baseline_preprocess(
    monkeypatch,
    tmp_path,
) -> None:
    import scripts.core_structure_ablation as ablation_module

    suite_dir = tmp_path / "suite"
    baseline_run_dir = suite_dir / "baseline"
    baseline_run_dir.mkdir(parents=True)
    for filename in ablation_module.SPLIT_ARTIFACT_FILES + ablation_module.PREPROCESS_ARTIFACT_FILES:
        (baseline_run_dir / filename).write_text(json.dumps({"stub": filename}), encoding="utf-8")
    for split in ablation_module.PRECOMPUTED_SPLITS:
        split_dir = baseline_run_dir / "precomputed" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "manifest.json").write_text(json.dumps({"split": split}), encoding="utf-8")

    observed = {}

    def fake_run_training_pipeline(run_dir, config, *, force_rebuild):
        del config
        observed["force_rebuild"] = force_rebuild
        observed["source_stats_exists"] = (run_dir / "source_stats.json").exists()
        observed["precomputed_exists"] = (run_dir / "precomputed" / "train" / "manifest.json").exists()
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "essence_forge_summary.json").write_text(
            json.dumps(
                {
                    "finetuned_model_target": {"macro_f1": 0.5, "accuracy": 0.5, "gmean": 0.5},
                    "finetuned_model_thresholded_target": {
                        "macro_f1": 0.6,
                        "accuracy": 0.6,
                        "gmean": 0.6,
                    },
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ablation_module, "_run_training_pipeline", fake_run_training_pipeline)

    result = run_experiment(
        base_config=load_experiment_config(),
        experiment={
            "id": "grouped_shared_e3",
            "description": "Grouped shared deform",
            "overrides": {
                "cross_sensor_residuals": {"enable": False},
                "model": {
                    "use_deformable_tcn": True,
                    "deform_scope": "raw_only",
                    "deform_bypass_health_mask": True,
                    "deform_group_mode": "shared_by_group",
                    "deform_groups": [
                        "accel",
                        "gyro",
                        "mag",
                        "pos",
                        "vel",
                        "quat",
                        "actuator_rpm",
                        "baro",
                    ],
                    "deform_apply_levels": [0],
                },
            },
        },
        suite_dir=suite_dir,
        base_fingerprint="mismatch-on-purpose",
        force_rebuild=True,
        skip_existing=False,
    )

    assert result["experiment_id"] == "grouped_shared_e3"
    assert result["reused_preprocess"] is True
    assert observed["force_rebuild"] is False
    assert observed["source_stats_exists"] is True
    assert observed["precomputed_exists"] is True


def test_copy_preprocess_artifacts_prefers_symlink_for_precomputed(tmp_path: Path) -> None:
    source_run_dir = tmp_path / "source"
    target_run_dir = tmp_path / "target"
    source_run_dir.mkdir()
    target_run_dir.mkdir()
    for filename in ("source_stats.json", "split_source_train.json"):
        (source_run_dir / filename).write_text(json.dumps({"stub": filename}), encoding="utf-8")

    source_precomputed = source_run_dir / "precomputed" / "train"
    source_precomputed.mkdir(parents=True)
    (source_precomputed / "manifest.json").write_text(json.dumps({"split": "train"}), encoding="utf-8")

    _copy_preprocess_artifacts(source_run_dir, target_run_dir)

    target_precomputed = target_run_dir / "precomputed"
    assert target_precomputed.is_symlink()
    assert os.readlink(target_precomputed) != ""
    assert (target_run_dir / "source_stats.json").exists()


def test_copy_preprocess_artifacts_falls_back_to_copy_when_symlink_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import scripts.core_structure_ablation as ablation_module

    source_run_dir = tmp_path / "source"
    target_run_dir = tmp_path / "target"
    source_run_dir.mkdir()
    target_run_dir.mkdir()
    for filename in ("source_stats.json", "split_source_train.json"):
        (source_run_dir / filename).write_text(json.dumps({"stub": filename}), encoding="utf-8")

    source_precomputed = source_run_dir / "precomputed" / "train"
    source_precomputed.mkdir(parents=True)
    (source_precomputed / "manifest.json").write_text(json.dumps({"split": "train"}), encoding="utf-8")

    def raise_symlink(*args, **kwargs):
        raise OSError("symlink blocked")

    monkeypatch.setattr(ablation_module.os, "symlink", raise_symlink)

    _copy_preprocess_artifacts(source_run_dir, target_run_dir)

    target_precomputed = target_run_dir / "precomputed"
    assert target_precomputed.is_symlink() is False
    assert (target_precomputed / "train" / "manifest.json").exists()
