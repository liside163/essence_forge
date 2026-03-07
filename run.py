from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

if __package__ in {None, ""}:
    _PACKAGE_DIR = Path(__file__).resolve().parent
    _PROJECT_DIR = _PACKAGE_DIR.parent
    if str(_PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_DIR))

from essence_forge.config import BUNDLED_CONFIG_PATH, ExperimentConfig, load_experiment_config
from essence_forge.core.runtime_config import CFG, reload_config
from essence_forge.preprocess import ensure_preprocess_artifacts
from essence_forge.split import build_index, save_splits, split_records


SOURCE_MODEL_NAME = "tcn_source_essence_forge"
TARGET_MODEL_NAME = "tcn_finetuned_essence_forge"


def _activate_runtime_config(config: ExperimentConfig, run_dir: Path) -> Path:
    snapshot_path = config.write_runtime_snapshot(run_dir)
    os.environ["ESSENCE_FORGE_CONFIG_PATH"] = str(snapshot_path)
    os.environ["UAV_TCN_CONFIG_PATH"] = str(snapshot_path)
    reload_config(snapshot_path)
    return snapshot_path


def _resolve_run_dir(config: ExperimentConfig, run_dir: str | Path | None) -> Path:
    if run_dir is None:
        return config.default_run_dir()
    return Path(run_dir).expanduser().resolve()


def cmd_split(run_dir: Path, config: ExperimentConfig) -> Dict[str, int]:
    _activate_runtime_config(config, run_dir)

    source_records = build_index(
        data_dir=Path(CFG.data_dir),
        use_A=CFG.source_A,
        use_B=CFG.source_B,
        use_faults=CFG.source_faults,
    )
    target_records = build_index(
        data_dir=Path(CFG.data_dir),
        use_A=CFG.target_A,
        use_B=CFG.target_B,
        use_faults=CFG.target_faults,
    )

    source_train, source_val, source_test = split_records(
        source_records,
        train_ratio=CFG.train_ratio,
        val_ratio=CFG.val_ratio,
        test_ratio=CFG.test_ratio,
        seed=CFG.split_seed,
        stratify_by_fault=CFG.stratify_by_fault,
    )
    target_train, target_val, target_test = split_records(
        target_records,
        train_ratio=CFG.train_ratio,
        val_ratio=CFG.val_ratio,
        test_ratio=CFG.test_ratio,
        seed=CFG.split_seed,
        stratify_by_fault=CFG.stratify_by_fault,
    )

    save_splits(
        run_dir=run_dir,
        source_train=source_train,
        source_val=source_val,
        source_test=source_test,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
    )
    return {
        "source_train": len(source_train),
        "source_val": len(source_val),
        "source_test": len(source_test),
        "target_train": len(target_train),
        "target_val": len(target_val),
        "target_test": len(target_test),
    }


def cmd_preprocess(
    run_dir: Path,
    config: ExperimentConfig,
    force_rebuild: bool = False,
) -> None:
    _activate_runtime_config(config, run_dir)
    ensure_preprocess_artifacts(
        run_dir=run_dir,
        config=config,
        force_rebuild=force_rebuild,
    )


def cmd_train(run_dir: Path, config: ExperimentConfig) -> str:
    _activate_runtime_config(config, run_dir)
    from essence_forge.core.train import train_source_model

    result = train_source_model(
        run_dir=run_dir,
        model_name=SOURCE_MODEL_NAME,
        model_class_name="essence_forge_tcn",
    )
    return result.checkpoint_path


def cmd_finetune(run_dir: Path, config: ExperimentConfig) -> str:
    _activate_runtime_config(config, run_dir)
    from essence_forge.core.fine_tune import fine_tune_model

    source_ckpt = run_dir / "checkpoints" / f"{SOURCE_MODEL_NAME}.pt"
    result = fine_tune_model(
        run_dir=run_dir,
        source_checkpoint=str(source_ckpt),
        model_name=TARGET_MODEL_NAME,
    )
    return result.checkpoint_path


def _build_summary(run_dir: Path) -> Dict[str, Any]:
    eval_dir = run_dir / "eval"
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
    }
    for filename in (
        "source_model_direct_target.json",
        "source_model_thresholded_target.json",
        "finetuned_model_target.json",
        "finetuned_model_thresholded_target.json",
        "transfer_gap_summary.json",
    ):
        path = eval_dir / filename
        if path.exists():
            summary[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return summary


def cmd_eval(run_dir: Path, config: ExperimentConfig) -> Dict[str, Any]:
    _activate_runtime_config(config, run_dir)
    from essence_forge.core.evaluate import evaluate_model, evaluate_transfer_gap

    source_ckpt = run_dir / "checkpoints" / f"{SOURCE_MODEL_NAME}.pt"
    target_ckpt = run_dir / "checkpoints" / f"{TARGET_MODEL_NAME}.pt"

    if source_ckpt.exists() and target_ckpt.exists():
        evaluate_transfer_gap(run_dir=run_dir, source_ckpt=str(source_ckpt), target_ckpt=str(target_ckpt))
    elif source_ckpt.exists():
        evaluate_model(run_dir=run_dir, checkpoint=str(source_ckpt), domain="source", model_name="source_model")
    elif target_ckpt.exists():
        evaluate_model(run_dir=run_dir, checkpoint=str(target_ckpt), domain="target", model_name="finetuned_model")
    else:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / 'checkpoints'}")

    summary = _build_summary(run_dir)
    summary_path = run_dir / "eval" / "essence_forge_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def cmd_pipeline(
    run_dir: Path,
    config: ExperimentConfig,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    _activate_runtime_config(config, run_dir)
    cmd_split(run_dir, config)
    cmd_preprocess(run_dir, config, force_rebuild=force_rebuild)
    cmd_train(run_dir, config)
    cmd_finetune(run_dir, config)
    return cmd_eval(run_dir, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Essence Forge experiment.")
    parser.add_argument(
        "command",
        choices=["split", "preprocess", "train", "finetune", "eval", "pipeline"],
    )
    parser.add_argument("--config", type=str, default=str(BUNDLED_CONFIG_PATH))
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_experiment_config(args.config)
    run_dir = _resolve_run_dir(config, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "split":
        cmd_split(run_dir, config)
        return 0
    if args.command == "preprocess":
        cmd_preprocess(run_dir, config, force_rebuild=bool(args.force_rebuild))
        return 0
    if args.command == "train":
        cmd_train(run_dir, config)
        return 0
    if args.command == "finetune":
        cmd_finetune(run_dir, config)
        return 0
    if args.command == "eval":
        cmd_eval(run_dir, config)
        return 0

    cmd_pipeline(run_dir, config, force_rebuild=bool(args.force_rebuild))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
