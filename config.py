from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
BUNDLED_CONFIG_ID = "b219593f866770c2c58f7c17b902a4b5d4573a5c5836beafd28c526d4dfaf078"
READABLE_EXPERIMENT_SLUG = "simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s"
BUNDLED_CONFIG_FILENAME = f"{READABLE_EXPERIMENT_SLUG}.json"
BUNDLED_CONFIG_PATH = PACKAGE_DIR / "configs" / BUNDLED_CONFIG_FILENAME
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs" / "essence_forge"
MODEL_CLASS_NAME = "essence_forge_tcn"
EXPERIMENT_NAME = "essence_forge"
EXPERIMENT_TITLE = "淬真 / Essence Forge: Distilling signal essence from noisy UAV telemetry"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    title: str
    config_path: Path
    payload: Dict[str, Any]
    bundle_id: str = BUNDLED_CONFIG_ID
    model_class_name: str = MODEL_CLASS_NAME

    @property
    def output_root(self) -> Path:
        return DEFAULT_OUTPUT_ROOT

    def default_run_dir(self) -> Path:
        return self.output_root / READABLE_EXPERIMENT_SLUG

    def runtime_payload(self) -> Dict[str, Any]:
        payload = copy.deepcopy(self.payload)
        payload.setdefault("paths", {})
        payload["paths"]["outputs_dir"] = str(self.output_root)
        payload.setdefault("essence_forge_meta", {})
        payload["essence_forge_meta"].update(
            {
                "name": self.name,
                "title": self.title,
                "bundle_id": self.bundle_id,
                "model_class_name": self.model_class_name,
                "bundled_config_path": str(self.config_path),
                "standalone_package_dir": str(PACKAGE_DIR),
            }
        )
        return payload

    def write_runtime_snapshot(self, run_dir: Path) -> Path:
        snapshot_path = run_dir / "configs" / "essence_forge.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("w", encoding="utf-8") as file:
            json.dump(self.runtime_payload(), file, ensure_ascii=False, indent=2)
        return snapshot_path


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    resolved = BUNDLED_CONFIG_PATH if path is None else Path(path).expanduser().resolve()
    payload = _read_json(resolved)
    return ExperimentConfig(
        name=EXPERIMENT_NAME,
        title=EXPERIMENT_TITLE,
        config_path=resolved,
        payload=payload,
    )
