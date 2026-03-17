from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np

from essence_forge.core.channel_layout import DEFAULT_30_RAW_CHANNEL_NAMES
from essence_forge.core.runtime_config import load_config

from essence_forge.core.cross_sensor_residuals import (
    SUPPORTED_CROSS_SENSOR_RESIDUAL_SCHEMES,
    compute_calibrated_cross_sensor_residuals,
    default_cross_sensor_residual_channel_names,
    fit_cross_sensor_residual_calibration,
)


DT = 1.0 / 120.0


def _quaternion_from_roll_pitch(roll: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    q0 = cr * cp
    q1 = sr * cp
    q2 = cr * sp
    q3 = -sr * sp
    q = np.stack([q0, q1, q2, q3], axis=1)
    norm = np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-8)
    return (q / norm).astype(np.float32)


def _build_consistent_raw_sequence(length: int = 128, mixer_lag: int = 1) -> np.ndarray:
    names = list(DEFAULT_30_RAW_CHANNEL_NAMES)
    idx = {name: i for i, name in enumerate(names)}
    t = np.arange(length, dtype=np.float32) * np.float32(DT)

    ctrl = np.stack(
        [
            0.12 * np.sin(2.0 * math.pi * 0.6 * t),
            0.11 * np.cos(2.0 * math.pi * 0.7 * t),
            0.08 * np.sin(2.0 * math.pi * 0.4 * t + 0.3),
            0.55 + 0.05 * np.cos(2.0 * math.pi * 0.5 * t),
        ],
        axis=1,
    ).astype(np.float32)
    mixer = np.asarray(
        [
            [-2900.0, 4000.0, 3100.0, 11550.0, 2230.0],
            [7050.0, -6840.0, 2970.0, 11460.0, 2260.0],
            [7140.0, 3110.0, -3800.0, 11480.0, 2255.0],
            [-2865.0, -5345.0, -3820.0, 11535.0, 2238.0],
        ],
        dtype=np.float32,
    )
    rpm = np.zeros((length, 4), dtype=np.float32)
    ctrl_aug = np.concatenate([ctrl, np.ones((length, 1), dtype=np.float32)], axis=1)
    rpm_base = ctrl_aug @ mixer.T
    rpm[mixer_lag:] = rpm_base[:-mixer_lag]
    rpm[:mixer_lag] = rpm_base[0]

    roll = 0.06 * np.sin(2.0 * math.pi * 0.8 * t)
    pitch = 0.05 * np.cos(2.0 * math.pi * 0.6 * t)
    quat = _quaternion_from_roll_pitch(roll=roll, pitch=pitch)
    qdot = np.gradient(quat, DT, axis=0)
    gyro_x = 0.40 * (2.0 * qdot[:, 1]) - 1e-3
    gyro_y = 0.40 * (2.0 * qdot[:, 2]) + 1e-4
    gyro_z = 0.002 * np.sin(2.0 * math.pi * 0.3 * t)

    pos_x = 0.10 * np.sin(2.0 * math.pi * 0.2 * t)
    pos_y = 0.12 * np.cos(2.0 * math.pi * 0.25 * t)
    pos_z = -6.0 - 0.4 * np.sin(2.0 * math.pi * 0.15 * t)
    pos = np.stack([pos_x, pos_y, pos_z], axis=1).astype(np.float32)
    vel = np.gradient(pos, DT, axis=0).astype(np.float32)
    accel = np.gradient(vel, DT, axis=0).astype(np.float32)

    mag = np.stack(
        [
            0.28 + 0.01 * np.sin(2.0 * math.pi * 0.1 * t),
            -0.04 + 0.01 * np.cos(2.0 * math.pi * 0.12 * t),
            0.47 + 0.01 * np.sin(2.0 * math.pi * 0.08 * t + 0.2),
        ],
        axis=1,
    ).astype(np.float32)
    baro_alt = (-1.0 * pos_z + 0.9).astype(np.float32)
    baro_temp = (24.9 + 0.03 * np.sin(2.0 * math.pi * 0.05 * t)).astype(np.float32)
    baro_pressure = (101242.0 + 50.0 * np.cos(2.0 * math.pi * 0.05 * t)).astype(np.float32)

    raw = np.zeros((length, len(names)), dtype=np.float32)
    raw[:, idx["accel_x"] : idx["accel_z"] + 1] = accel
    raw[:, idx["gyro_x"] : idx["gyro_z"] + 1] = np.stack([gyro_x, gyro_y, gyro_z], axis=1)
    raw[:, idx["mag_x"] : idx["mag_z"] + 1] = mag
    raw[:, idx["pos_x"] : idx["pos_z"] + 1] = pos
    raw[:, idx["vel_x"] : idx["vel_z"] + 1] = vel
    raw[:, idx["q0"] : idx["q3"] + 1] = quat
    raw[:, idx["actuator_ctrl_0"] : idx["actuator_ctrl_3"] + 1] = ctrl
    raw[:, idx["motor_rpm_1"] : idx["motor_rpm_4"] + 1] = rpm
    raw[:, idx["baro_alt"]] = baro_alt
    raw[:, idx["baro_temp"]] = baro_temp
    raw[:, idx["baro_pressure"]] = baro_pressure
    return raw


def test_modal_baro_att8_scheme_is_available() -> None:
    assert "modal_baro_att8" in SUPPORTED_CROSS_SENSOR_RESIDUAL_SCHEMES
    assert default_cross_sensor_residual_channel_names("modal_baro_att8") == (
        "rpm_modal_common_res",
        "rpm_modal_a_res",
        "rpm_modal_b_res",
        "rpm_modal_c_res",
        "baro_pos_res",
        "baro_vel_res",
        "qgyro_x_res",
        "qgyro_y_res",
    )


def test_modal_baro_att8_calibration_and_generation_work() -> None:
    raw = _build_consistent_raw_sequence(length=160, mixer_lag=1)
    fit_payload = fit_cross_sensor_residual_calibration(
        raw_sequences=[raw],
        sample_rate_hz=120.0,
        max_lag_steps=4,
    )
    residuals = compute_calibrated_cross_sensor_residuals(
        raw=raw,
        fit_payload=fit_payload,
        scheme="modal_baro_att8",
        clip_value=6.0,
    )

    assert fit_payload["rpm_mixer"]["lag"] == 1
    assert residuals.shape == (160, 8)
    assert np.isfinite(residuals).all()


def test_runtime_config_ignores_stale_legacy_channels_when_switching_to_modal_att8(
    tmp_path: Path,
) -> None:
    config_path = Path("configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["cross_sensor_residuals"]["enable"] = True
    payload["cross_sensor_residuals"]["scheme"] = "modal_baro_att8"
    payload["cross_sensor_residuals"]["clip_value"] = 6.0
    payload["cross_sensor_residuals"]["calibration_split"] = "source_train_nofault"
    payload["cross_sensor_residuals"]["max_lag_steps"] = 4
    assert len(payload["cross_sensor_residuals"]["channels"]) == 9

    custom_path = tmp_path / "config.json"
    custom_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = load_config(custom_path)

    assert cfg.cross_sensor_residual_scheme == "modal_baro_att8"
    assert cfg.cross_sensor_residual_channels == 8
    assert cfg.cross_sensor_residual_channel_names == default_cross_sensor_residual_channel_names(
        "modal_baro_att8"
    )
