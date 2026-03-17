# E4 Scale Wide Default Design

## Context

The repository already contains the `e4_scale_wide` ablation configuration and the supporting model/runtime code for bounded grouped deformable alignment. However, the default entrypoint configs still describe the older default behavior, and `README.md` still documents deprecated cross-sensor residual features as part of the default pipeline.

The user wants all default entrypoints to align with the `e4_scale_wide` experiment, residual features removed from default-facing documentation, and the README updated to describe the current deformable-convolution design.

## Goal

Make the repository defaults match the best supported `e4_scale_wide` configuration and update documentation so the default behavior described in the repo is accurate.

## Scope

- Update all default config entrypoints to the `e4_scale_wide` parameter set.
- Remove deprecated residual-feature language from the README's default-pipeline description.
- Rewrite the deformable-convolution description to reflect the current bounded grouped setup.
- Keep ablation configs and code support for residual utilities intact unless they are part of default-facing docs or config paths.

## Non-Goals

- Re-running training or changing saved experiment outputs.
- Removing residual-support code from the repository.
- Changing the ablation suite structure or experiment ranking logic.

## Recommended Approach

Apply the `e4_scale_wide` override values directly to both default config JSON files:

- `configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json`
- `configs/b219593f866770c2c58f7c17b902a4b5d4573a5c5836beafd28c526d4dfaf078.json`

This keeps the default runtime path explicit and inspectable. It is preferable to adding hidden indirection in `run.py` or asking users to mentally combine a base config with an ablation override.

For docs, update `README.md` so it reflects:

- default inputs are the 30 base channels plus health-mask augmentation, not residual channels;
- cross-sensor residuals are not part of the default experiment path;
- the default deformable alignment is bounded, grouped, raw-only, health-mask-bypassed, applied at level 0, and uses the widened offset scale.

## Expected Config Changes

Default model parameters should match `configs/ablations/e4_boost.json` experiment `e4_scale_wide`:

- `cross_sensor_residuals.enable = false`
- `model.use_deformable_tcn = true`
- `model.concat_health_mask_channels = true`
- `model.deform_scope = "raw_only"`
- `model.deform_bypass_health_mask = true`
- `model.deform_group_mode = "shared_by_group"`
- `model.deform_groups = ["accel", "gyro", "mag", "pos", "vel", "quat", "actuator_rpm", "baro"]`
- `model.deform_apply_levels = [0]`
- `model.deform_max_offset_scale = 1.5`
- `model.deform_zero_init = true`
- `model.deform_warmup_epochs = 5`
- `model.deform_offset_lr_scale = 0.1`
- `model.deform_offset_l1_weight = 0.0005`
- `model.deform_offset_tv_weight = 0.001`
- `model.use_deform_conv_gate = false`

## Documentation Changes

`README.md` should be corrected in three places:

1. Input-feature description
   - Remove statements that default data construction appends cross-sensor residual features.
   - Describe health masks as the only default auxiliary signal.

2. Method description
   - Replace the old generic deformable-convolution wording with the current bounded grouped alignment wording.

3. Preprocess / repo-structure / usage sections
   - Remove mentions that residual features are a default preprocessing output or default model input.

## Verification

Verification should be run under `WSL + conda torch128` and should cover the tests already added for:

- ablation runner behavior
- deformable redesign behavior
- frequency branch behavior
- cross-sensor residual helper behavior
- preprocess cleanup behavior

The verification target is to prove that switching default config values and README text does not break config loading or the existing regression tests.
