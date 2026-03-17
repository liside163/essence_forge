# E4 Scale Wide Default Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Switch all default config entrypoints to the `e4_scale_wide` parameter set and update README documentation to remove deprecated residual-feature language and describe the current deformable-alignment design accurately.

**Architecture:** The change is configuration-first. Update the two default config JSON files so the repository's direct entrypoints encode the `e4_scale_wide` settings explicitly, then update `README.md` so the documented default pipeline matches those configs. Verification remains at the regression-test level because the change does not alter training code paths beyond default parameter selection.

**Tech Stack:** Python project, JSON configs, Markdown docs, pytest under WSL with conda `torch128`.

---

### Task 1: Align Default Config Entrypoints

**Files:**
- Modify: `D:/Bigshe/TFL/UAV_TCN_Fine-turn/essence_forge/configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json`
- Modify: `D:/Bigshe/TFL/UAV_TCN_Fine-turn/essence_forge/configs/b219593f866770c2c58f7c17b902a4b5d4573a5c5836beafd28c526d4dfaf078.json`
- Reference: `D:/Bigshe/TFL/UAV_TCN_Fine-turn/essence_forge/configs/ablations/e4_boost.json`

**Step 1: Confirm the exact override set**

Read the `e4_scale_wide` experiment in `configs/ablations/e4_boost.json` and extract the model and residual override values.

**Step 2: Apply the default config changes**

Set both default configs to match the `e4_scale_wide` defaults:

- disable residual features for the default path
- enable raw-only grouped deformable alignment
- widen offset scale to `1.5`
- apply deformable alignment only at level `0`
- keep health-mask concatenation and bypass behavior aligned with the ablation config

**Step 3: Verify config diff**

Run:

```bash
git diff -- configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json configs/b219593f866770c2c58f7c17b902a4b5d4573a5c5836beafd28c526d4dfaf078.json
```

Expected: only the intended default-parameter changes appear.

### Task 2: Update README Default-Pipeline Documentation

**Files:**
- Modify: `D:/Bigshe/TFL/UAV_TCN_Fine-turn/essence_forge/README.md`

**Step 1: Remove deprecated residual-feature descriptions**

Delete or rewrite README passages that claim default preprocessing or default model inputs include cross-sensor residual channels.

**Step 2: Rewrite default-method description**

Describe the current default model as:

- 30 base channels plus health-mask augmentation
- time branch plus optional frequency branch
- bounded grouped deformable alignment
- raw-only deformable path with health-mask bypass
- level-0 grouped offsets with widened bounded range

**Step 3: Check for stale terminology**

Search for lingering default-facing residual references and update them if they imply residuals are still part of the default path.

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path("README.md").read_text(encoding="utf-8")
for needle in ["跨传感器残差", "残差特征", "9 个跨传感器残差通道"]:
    print(needle, text.count(needle))
PY
```

Expected: remaining matches, if any, are only acceptable non-default contextual mentions.

### Task 3: Regression Verification

**Files:**
- Verify: `D:/Bigshe/TFL/UAV_TCN_Fine-turn/essence_forge/tests/test_ablation_core_runner.py`
- Verify: `D:/Bigshe/TFL/UAV_TCN_Fine-turn/essence_forge/tests/test_cross_sensor_residuals.py`
- Verify: `D:/Bigshe/TFL/UAV_TCN_Fine-turn/essence_forge/tests/test_deformable_redesign.py`
- Verify: `D:/Bigshe/TFL/UAV_TCN_Fine-turn/essence_forge/tests/test_freq_branch_model.py`
- Verify: `D:/Bigshe/TFL/UAV_TCN_Fine-turn/essence_forge/tests/test_preprocess_cleanup.py`

**Step 1: Run targeted regression tests**

Run:

```bash
wsl bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch128 && cd /mnt/d/Bigshe/TFL/UAV_TCN_Fine-turn/essence_forge && python -m pytest tests/test_ablation_core_runner.py tests/test_cross_sensor_residuals.py tests/test_deformable_redesign.py tests/test_freq_branch_model.py tests/test_preprocess_cleanup.py -q"
```

Expected: all tests pass.

**Step 2: Inspect final diff**

Run:

```bash
git diff --stat
```

Expected: only the two config files and `README.md` change for this task.

### Task 4: Commit and Publish

**Files:**
- Commit the validated config/doc changes

**Step 1: Commit**

Run:

```bash
git add README.md configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json configs/b219593f866770c2c58f7c17b902a4b5d4573a5c5836beafd28c526d4dfaf078.json
git commit -m "docs: align defaults with e4 scale wide"
```

**Step 2: Push**

Run:

```bash
git push origin main
```

Expected: remote `main` updated with the default-config alignment commit.
