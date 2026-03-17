from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from essence_forge.core.channel_layout import (
    DEFAULT_30_RAW_CHANNEL_NAMES,
    LEGACY_ACCEL_INDICES,
    LEGACY_CORE_CHANNEL_COUNT,
    LEGACY_POS_INDICES,
    LEGACY_VEL_INDICES,
)


LEGACY9_SCHEME = "legacy9"
MODAL_BARO_VERT7_SCHEME = "modal_baro_vert7"
MODAL_BARO_ATT8_SCHEME = "modal_baro_att8"
MODAL_BARO_ATT_VERT9_SCHEME = "modal_baro_att_vert9"

SUPPORTED_CROSS_SENSOR_RESIDUAL_SCHEMES: tuple[str, ...] = (
    LEGACY9_SCHEME,
    MODAL_BARO_VERT7_SCHEME,
    MODAL_BARO_ATT8_SCHEME,
    MODAL_BARO_ATT_VERT9_SCHEME,
)

LEGACY9_CHANNEL_NAMES: tuple[str, ...] = (
    "imu_ekf_vel_res_x",
    "imu_ekf_vel_res_y",
    "imu_ekf_vel_res_z",
    "pos_vel_res_x",
    "pos_vel_res_y",
    "pos_vel_res_z",
    "accel_pos_res_x",
    "accel_pos_res_y",
    "accel_pos_res_z",
)

MODAL_BARO_VERT7_CHANNEL_NAMES: tuple[str, ...] = (
    "rpm_modal_common_res",
    "rpm_modal_a_res",
    "rpm_modal_b_res",
    "rpm_modal_c_res",
    "baro_pos_res",
    "baro_vel_res",
    "acc_vel_z_res",
)

MODAL_BARO_ATT8_CHANNEL_NAMES: tuple[str, ...] = (
    "rpm_modal_common_res",
    "rpm_modal_a_res",
    "rpm_modal_b_res",
    "rpm_modal_c_res",
    "baro_pos_res",
    "baro_vel_res",
    "qgyro_x_res",
    "qgyro_y_res",
)

MODAL_BARO_ATT_VERT9_CHANNEL_NAMES: tuple[str, ...] = (
    "rpm_modal_common_res",
    "rpm_modal_a_res",
    "rpm_modal_b_res",
    "rpm_modal_c_res",
    "baro_pos_res",
    "baro_vel_res",
    "qgyro_x_res",
    "qgyro_y_res",
    "acc_vel_z_res",
)

_SCHEME_TO_CHANNEL_NAMES: dict[str, tuple[str, ...]] = {
    LEGACY9_SCHEME: LEGACY9_CHANNEL_NAMES,
    MODAL_BARO_VERT7_SCHEME: MODAL_BARO_VERT7_CHANNEL_NAMES,
    MODAL_BARO_ATT8_SCHEME: MODAL_BARO_ATT8_CHANNEL_NAMES,
    MODAL_BARO_ATT_VERT9_SCHEME: MODAL_BARO_ATT_VERT9_CHANNEL_NAMES,
}

_RPM_MODAL_BASIS = np.asarray(
    [
        [0.5, 0.5, 0.5, 0.5],
        [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0, 0.0],
        [0.0, 0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
        [0.5, 0.5, -0.5, -0.5],
    ],
    dtype=np.float64,
).T

_DEFAULT_CALIBRATION_SPLIT = "source_train_nofault"


def normalize_cross_sensor_residual_scheme(scheme: str | None) -> str:
    raw = str(scheme or LEGACY9_SCHEME).strip().lower()
    if raw not in _SCHEME_TO_CHANNEL_NAMES:
        supported = ", ".join(SUPPORTED_CROSS_SENSOR_RESIDUAL_SCHEMES)
        raise ValueError(f"Unsupported cross_sensor_residuals.scheme={scheme!r}; supported={supported}")
    return raw


def default_cross_sensor_residual_channel_names(scheme: str | None) -> tuple[str, ...]:
    normalized = normalize_cross_sensor_residual_scheme(scheme)
    return _SCHEME_TO_CHANNEL_NAMES[normalized]


def cross_sensor_residual_channel_count_for_scheme(scheme: str | None) -> int:
    return len(default_cross_sensor_residual_channel_names(scheme))


def is_calibrated_cross_sensor_residual_scheme(scheme: str | None) -> bool:
    return normalize_cross_sensor_residual_scheme(scheme) != LEGACY9_SCHEME


def required_raw_channel_names_for_scheme(scheme: str | None) -> tuple[str, ...]:
    normalized = normalize_cross_sensor_residual_scheme(scheme)
    if normalized == LEGACY9_SCHEME:
        return (
            "accel_x",
            "accel_y",
            "accel_z",
            "pos_x",
            "pos_y",
            "pos_z",
            "vel_x",
            "vel_y",
            "vel_z",
        )
    return (
        "actuator_ctrl_0",
        "actuator_ctrl_1",
        "actuator_ctrl_2",
        "actuator_ctrl_3",
        "motor_rpm_1",
        "motor_rpm_2",
        "motor_rpm_3",
        "motor_rpm_4",
        "pos_z",
        "vel_z",
        "accel_z",
        "baro_alt",
        "gyro_x",
        "gyro_y",
        "q0",
        "q1",
        "q2",
        "q3",
    )


def default_cross_sensor_residual_config() -> dict[str, Any]:
    return {
        "scheme": LEGACY9_SCHEME,
        "clip_value": 6.0,
        "calibration_split": _DEFAULT_CALIBRATION_SPLIT,
        "max_lag_steps": 4,
    }


def _name_to_index(channel_names: Sequence[str]) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(channel_names)}


def _require_indices(channel_names: Sequence[str], required_names: Sequence[str]) -> dict[str, int]:
    mapping = _name_to_index(channel_names)
    missing = [name for name in required_names if name not in mapping]
    if missing:
        raise ValueError(f"Missing required channels for residual computation: {missing}")
    return mapping


def _as_float_array(values: Sequence[float], *, ndim: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"Expected array ndim={ndim}, actual={array.ndim}")
    return array


def compute_legacy_cross_sensor_residuals(
    raw: np.ndarray,
    dt: float = 1.0 / 120.0,
    normalize: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=np.float32)
    if raw_arr.ndim != 2:
        raise ValueError(f"raw must be a 2D array, actual={raw_arr.ndim}D")
    if raw_arr.shape[1] < LEGACY_CORE_CHANNEL_COUNT:
        raise ValueError(
            f"raw must include at least {LEGACY_CORE_CHANNEL_COUNT} channels, actual={raw_arr.shape[1]}"
        )
    if float(dt) <= 0.0:
        raise ValueError(f"dt must be > 0, actual={dt}")

    accel = raw_arr[:, list(LEGACY_ACCEL_INDICES)]
    vel = raw_arr[:, list(LEGACY_VEL_INDICES)]
    pos = raw_arr[:, list(LEGACY_POS_INDICES)]

    residuals = np.zeros((raw_arr.shape[0], 9), dtype=np.float32)
    dvel = np.diff(vel, axis=0, prepend=vel[:1]) / float(dt)
    residuals[:, 0:3] = dvel - accel
    dpos = np.diff(pos, axis=0, prepend=pos[:1]) / float(dt)
    residuals[:, 3:6] = dpos - vel
    ddpos = np.diff(dpos, axis=0, prepend=dpos[:1]) / float(dt)
    residuals[:, 6:9] = ddpos - accel

    if bool(normalize):
        scale = np.std(residuals, axis=0, dtype=np.float32)
        scale = np.maximum(scale, float(eps)).astype(np.float32)
        residuals = residuals / scale.reshape(1, -1)
    return residuals.astype(np.float32, copy=False)


def _gradient(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(values, float(dt), axis=0).astype(np.float64, copy=False)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    return quaternion / np.maximum(norm, 1e-8)


def _quaternion_to_omega(quaternion: np.ndarray, dt: float) -> np.ndarray:
    q = _normalize_quaternion(np.asarray(quaternion, dtype=np.float64))
    qdot = _gradient(q, dt)
    omega = np.zeros((q.shape[0], 3), dtype=np.float64)
    for t in range(q.shape[0]):
        q0, q1, q2, q3 = q[t]
        matrix = np.asarray(
            [
                [-q1, -q2, -q3],
                [q0, -q3, q2],
                [q3, q0, -q1],
                [-q2, q1, q0],
            ],
            dtype=np.float64,
        )
        omega[t], *_ = np.linalg.lstsq(matrix, 2.0 * qdot[t], rcond=None)
    return omega


def _fit_linear_relation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    design = np.stack([x_vec, np.ones_like(x_vec)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, y_vec, rcond=None)
    pred = design @ coeffs
    corr = 0.0
    if float(np.std(x_vec)) > 1e-8 and float(np.std(y_vec)) > 1e-8:
        corr = float(np.corrcoef(x_vec, y_vec)[0, 1])
    sigma = float(np.std(y_vec - pred) + 1e-8)
    return {
        "a": float(coeffs[0]),
        "b": float(coeffs[1]),
        "corr": corr,
        "sigma": sigma,
    }


def _fit_rpm_mixer(raw_sequences: Sequence[np.ndarray], channel_names: Sequence[str], max_lag_steps: int) -> dict[str, Any]:
    mapping = _require_indices(
        channel_names,
        (
            "actuator_ctrl_0",
            "actuator_ctrl_1",
            "actuator_ctrl_2",
            "actuator_ctrl_3",
            "motor_rpm_1",
            "motor_rpm_2",
            "motor_rpm_3",
            "motor_rpm_4",
        ),
    )
    ctrl_indices = [mapping[f"actuator_ctrl_{idx}"] for idx in range(4)]
    rpm_indices = [mapping[f"motor_rpm_{idx}"] for idx in range(1, 5)]

    best_fit: dict[str, Any] | None = None
    for lag in range(max(0, int(max_lag_steps)) + 1):
        ctrl_parts: list[np.ndarray] = []
        rpm_parts: list[np.ndarray] = []
        for raw in raw_sequences:
            ctrl = np.asarray(raw[:, ctrl_indices], dtype=np.float64)
            rpm = np.asarray(raw[:, rpm_indices], dtype=np.float64)
            if lag > 0:
                if ctrl.shape[0] <= lag:
                    continue
                ctrl = ctrl[:-lag]
                rpm = rpm[lag:]
            ctrl_parts.append(ctrl)
            rpm_parts.append(rpm)

        if not ctrl_parts:
            continue

        ctrl_all = np.concatenate(ctrl_parts, axis=0)
        rpm_all = np.concatenate(rpm_parts, axis=0)
        design = np.concatenate(
            [ctrl_all, np.ones((ctrl_all.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        weights, *_ = np.linalg.lstsq(design, rpm_all, rcond=None)
        pred = design @ weights
        correlations = []
        for motor_idx in range(rpm_all.shape[1]):
            if float(np.std(pred[:, motor_idx])) <= 1e-8 or float(np.std(rpm_all[:, motor_idx])) <= 1e-8:
                correlations.append(0.0)
                continue
            correlations.append(float(np.corrcoef(rpm_all[:, motor_idx], pred[:, motor_idx])[0, 1]))

        candidate = {
            "lag": int(lag),
            "weights": weights.tolist(),
            "corr": correlations,
            "score": float(np.mean(np.abs(np.asarray(correlations, dtype=np.float64)))),
        }
        if best_fit is None or candidate["score"] > best_fit["score"]:
            best_fit = candidate

    if best_fit is None:
        raise ValueError("Unable to fit RPM mixer residual calibration from empty inputs")

    residual_parts: list[np.ndarray] = []
    for raw in raw_sequences:
        residual_parts.append(
            _compute_rpm_modal_residuals(
                raw=raw,
                channel_names=channel_names,
                mixer_fit=best_fit,
                sigma=np.ones((4,), dtype=np.float64),
                clip_value=None,
            )[0]
        )
    sigma = np.std(np.concatenate(residual_parts, axis=0), axis=0)
    best_fit["sigma"] = np.maximum(sigma, 1e-8).astype(np.float64).tolist()
    best_fit.pop("score", None)
    return best_fit


def _fit_baro_relation(raw_sequences: Sequence[np.ndarray], channel_names: Sequence[str], dt: float) -> tuple[dict[str, float], dict[str, float]]:
    mapping = _require_indices(
        channel_names,
        (
            "pos_z",
            "vel_z",
            "baro_alt",
        ),
    )
    pos_z_idx = mapping["pos_z"]
    vel_z_idx = mapping["vel_z"]
    baro_alt_idx = mapping["baro_alt"]

    best_pos: dict[str, float] | None = None
    best_vel: dict[str, float] | None = None

    for sign in (1.0, -1.0):
        pos_inputs = np.concatenate([sign * np.asarray(raw[:, pos_z_idx], dtype=np.float64) for raw in raw_sequences])
        baro_alt = np.concatenate([np.asarray(raw[:, baro_alt_idx], dtype=np.float64) for raw in raw_sequences])
        pos_fit = _fit_linear_relation(pos_inputs, baro_alt)
        pos_fit["sign"] = float(sign)
        if best_pos is None or abs(pos_fit["corr"]) > abs(best_pos["corr"]):
            best_pos = pos_fit

        vel_inputs = np.concatenate([sign * np.asarray(raw[:, vel_z_idx], dtype=np.float64) for raw in raw_sequences])
        baro_vel = np.concatenate([_gradient(np.asarray(raw[:, [baro_alt_idx]], dtype=np.float64), dt)[:, 0] for raw in raw_sequences])
        vel_fit = _fit_linear_relation(vel_inputs, baro_vel)
        vel_fit["sign"] = float(sign)
        if best_vel is None or abs(vel_fit["corr"]) > abs(best_vel["corr"]):
            best_vel = vel_fit

    if best_pos is None or best_vel is None:
        raise ValueError("Unable to fit barometer calibration")
    return best_pos, best_vel


def _fit_qgyro_xy(raw_sequences: Sequence[np.ndarray], channel_names: Sequence[str], dt: float) -> list[dict[str, float]]:
    mapping = _require_indices(
        channel_names,
        (
            "q0",
            "q1",
            "q2",
            "q3",
            "gyro_x",
            "gyro_y",
        ),
    )
    quaternion_indices = [mapping[f"q{idx}"] for idx in range(4)]
    gyro_indices = [mapping["gyro_x"], mapping["gyro_y"]]
    omega_x_parts: list[np.ndarray] = []
    omega_y_parts: list[np.ndarray] = []
    gyro_x_parts: list[np.ndarray] = []
    gyro_y_parts: list[np.ndarray] = []
    for raw in raw_sequences:
        quaternion = np.asarray(raw[:, quaternion_indices], dtype=np.float64)
        omega = _quaternion_to_omega(quaternion=quaternion, dt=dt)
        omega_x_parts.append(omega[:, 0])
        omega_y_parts.append(omega[:, 1])
        gyro_x_parts.append(np.asarray(raw[:, gyro_indices[0]], dtype=np.float64))
        gyro_y_parts.append(np.asarray(raw[:, gyro_indices[1]], dtype=np.float64))

    return [
        _fit_linear_relation(np.concatenate(omega_x_parts), np.concatenate(gyro_x_parts)),
        _fit_linear_relation(np.concatenate(omega_y_parts), np.concatenate(gyro_y_parts)),
    ]


def _fit_accel_vel_z(raw_sequences: Sequence[np.ndarray], channel_names: Sequence[str], dt: float) -> dict[str, float]:
    mapping = _require_indices(channel_names, ("accel_z", "vel_z"))
    accel_z = np.concatenate([np.asarray(raw[:, mapping["accel_z"]], dtype=np.float64) for raw in raw_sequences])
    dvel_z = np.concatenate(
        [
            _gradient(np.asarray(raw[:, [mapping["vel_z"]]], dtype=np.float64), dt)[:, 0]
            for raw in raw_sequences
        ]
    )
    return _fit_linear_relation(accel_z, dvel_z)


def fit_cross_sensor_residual_calibration(
    raw_sequences: Sequence[np.ndarray],
    sample_rate_hz: float = 120.0,
    max_lag_steps: int = 4,
    channel_names: Sequence[str] = DEFAULT_30_RAW_CHANNEL_NAMES,
) -> dict[str, Any]:
    sequences = [np.asarray(raw, dtype=np.float64) for raw in raw_sequences if np.asarray(raw).ndim == 2]
    if not sequences:
        raise ValueError("raw_sequences must include at least one [T, C] sequence")
    dt = 1.0 / max(float(sample_rate_hz), 1e-6)

    baro_pos_fit, baro_vel_fit = _fit_baro_relation(
        raw_sequences=sequences,
        channel_names=channel_names,
        dt=dt,
    )

    return {
        "version": 1,
        "sample_rate_hz": float(sample_rate_hz),
        "max_lag_steps": int(max_lag_steps),
        "channel_names": [str(name) for name in channel_names],
        "rpm_mixer": _fit_rpm_mixer(
            raw_sequences=sequences,
            channel_names=channel_names,
            max_lag_steps=max_lag_steps,
        ),
        "baro_pos": baro_pos_fit,
        "baro_vel": baro_vel_fit,
        "qgyro_xy": _fit_qgyro_xy(
            raw_sequences=sequences,
            channel_names=channel_names,
            dt=dt,
        ),
        "accel_vel_z": _fit_accel_vel_z(
            raw_sequences=sequences,
            channel_names=channel_names,
            dt=dt,
        ),
    }


def _compute_rpm_modal_residuals(
    raw: np.ndarray,
    channel_names: Sequence[str],
    mixer_fit: Mapping[str, Any],
    sigma: np.ndarray,
    clip_value: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    mapping = _require_indices(
        channel_names,
        (
            "actuator_ctrl_0",
            "actuator_ctrl_1",
            "actuator_ctrl_2",
            "actuator_ctrl_3",
            "motor_rpm_1",
            "motor_rpm_2",
            "motor_rpm_3",
            "motor_rpm_4",
        ),
    )
    ctrl_indices = [mapping[f"actuator_ctrl_{idx}"] for idx in range(4)]
    rpm_indices = [mapping[f"motor_rpm_{idx}"] for idx in range(1, 5)]
    ctrl = np.asarray(raw[:, ctrl_indices], dtype=np.float64)
    rpm = np.asarray(raw[:, rpm_indices], dtype=np.float64)
    weights = np.asarray(mixer_fit["weights"], dtype=np.float64)
    lag = int(mixer_fit["lag"])

    if lag > 0:
        design = np.concatenate(
            [ctrl[:-lag], np.ones((ctrl.shape[0] - lag, 1), dtype=np.float64)],
            axis=1,
        )
        pred = design @ weights
        residual = rpm[lag:] - pred
        residual = np.concatenate([np.repeat(residual[:1], lag, axis=0), residual], axis=0)
    else:
        design = np.concatenate([ctrl, np.ones((ctrl.shape[0], 1), dtype=np.float64)], axis=1)
        pred = design @ weights
        residual = rpm - pred

    modal = residual @ _RPM_MODAL_BASIS
    standardized = modal / np.maximum(sigma.reshape(1, -1), 1e-8)
    if clip_value is not None:
        standardized = np.clip(standardized, -float(clip_value), float(clip_value))
    return modal, standardized.astype(np.float32, copy=False)


def _compute_baro_residual(
    raw: np.ndarray,
    channel_names: Sequence[str],
    fit: Mapping[str, Any],
    clip_value: float | None,
    *,
    mode: str,
    dt: float,
) -> np.ndarray:
    mapping = _require_indices(channel_names, ("pos_z", "vel_z", "baro_alt"))
    if mode == "pos":
        predictor = float(fit["a"]) * (float(fit["sign"]) * np.asarray(raw[:, mapping["pos_z"]], dtype=np.float64)) + float(fit["b"])
        residual = np.asarray(raw[:, mapping["baro_alt"]], dtype=np.float64) - predictor
    else:
        predictor = float(fit["a"]) * (float(fit["sign"]) * np.asarray(raw[:, mapping["vel_z"]], dtype=np.float64)) + float(fit["b"])
        baro_vel = _gradient(np.asarray(raw[:, [mapping["baro_alt"]]], dtype=np.float64), dt)[:, 0]
        residual = baro_vel - predictor
    standardized = residual / max(float(fit["sigma"]), 1e-8)
    if clip_value is not None:
        standardized = np.clip(standardized, -float(clip_value), float(clip_value))
    return standardized.reshape(-1, 1).astype(np.float32, copy=False)


def _compute_qgyro_residual(
    raw: np.ndarray,
    channel_names: Sequence[str],
    fit: Mapping[str, Any],
    clip_value: float | None,
    *,
    axis: int,
    dt: float,
) -> np.ndarray:
    mapping = _require_indices(channel_names, ("q0", "q1", "q2", "q3", "gyro_x", "gyro_y"))
    quaternion_indices = [mapping[f"q{idx}"] for idx in range(4)]
    quaternion = np.asarray(raw[:, quaternion_indices], dtype=np.float64)
    omega = _quaternion_to_omega(quaternion=quaternion, dt=dt)
    gyro_name = "gyro_x" if axis == 0 else "gyro_y"
    gyro = np.asarray(raw[:, mapping[gyro_name]], dtype=np.float64)
    residual = gyro - (float(fit["a"]) * omega[:, axis] + float(fit["b"]))
    standardized = residual / max(float(fit["sigma"]), 1e-8)
    if clip_value is not None:
        standardized = np.clip(standardized, -float(clip_value), float(clip_value))
    return standardized.reshape(-1, 1).astype(np.float32, copy=False)


def _compute_accel_vel_z_residual(
    raw: np.ndarray,
    channel_names: Sequence[str],
    fit: Mapping[str, Any],
    clip_value: float | None,
    dt: float,
) -> np.ndarray:
    mapping = _require_indices(channel_names, ("accel_z", "vel_z"))
    accel_z = np.asarray(raw[:, mapping["accel_z"]], dtype=np.float64)
    dvel_z = _gradient(np.asarray(raw[:, [mapping["vel_z"]]], dtype=np.float64), dt)[:, 0]
    residual = dvel_z - (float(fit["a"]) * accel_z + float(fit["b"]))
    standardized = residual / max(float(fit["sigma"]), 1e-8)
    if clip_value is not None:
        standardized = np.clip(standardized, -float(clip_value), float(clip_value))
    return standardized.reshape(-1, 1).astype(np.float32, copy=False)


def compute_calibrated_cross_sensor_residuals(
    raw: np.ndarray,
    fit_payload: Mapping[str, Any],
    scheme: str | None,
    clip_value: float = 6.0,
) -> np.ndarray:
    normalized_scheme = normalize_cross_sensor_residual_scheme(scheme)
    if normalized_scheme == LEGACY9_SCHEME:
        dt = 1.0 / max(float(fit_payload.get("sample_rate_hz", 120.0)), 1e-6)
        return compute_legacy_cross_sensor_residuals(raw=raw, dt=dt, normalize=True, eps=1e-6)

    channel_names = tuple(str(name) for name in fit_payload.get("channel_names", DEFAULT_30_RAW_CHANNEL_NAMES))
    dt = 1.0 / max(float(fit_payload.get("sample_rate_hz", 120.0)), 1e-6)
    sigma = _as_float_array(fit_payload["rpm_mixer"]["sigma"], ndim=1)
    _, rpm_modal = _compute_rpm_modal_residuals(
        raw=raw,
        channel_names=channel_names,
        mixer_fit=fit_payload["rpm_mixer"],
        sigma=sigma,
        clip_value=clip_value,
    )
    features = [rpm_modal]
    features.append(_compute_baro_residual(raw, channel_names, fit_payload["baro_pos"], clip_value, mode="pos", dt=dt))
    features.append(_compute_baro_residual(raw, channel_names, fit_payload["baro_vel"], clip_value, mode="vel", dt=dt))
    if normalized_scheme in {MODAL_BARO_ATT8_SCHEME, MODAL_BARO_ATT_VERT9_SCHEME}:
        features.append(_compute_qgyro_residual(raw, channel_names, fit_payload["qgyro_xy"][0], clip_value, axis=0, dt=dt))
        features.append(_compute_qgyro_residual(raw, channel_names, fit_payload["qgyro_xy"][1], clip_value, axis=1, dt=dt))
    if normalized_scheme in {MODAL_BARO_VERT7_SCHEME, MODAL_BARO_ATT_VERT9_SCHEME}:
        features.append(
            _compute_accel_vel_z_residual(
                raw=raw,
                channel_names=channel_names,
                fit=fit_payload["accel_vel_z"],
                clip_value=clip_value,
                dt=dt,
            )
        )
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def serialize_cross_sensor_residual_fit(fit_payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(fit_payload)))


def write_cross_sensor_residual_fit(path: str | Path, fit_payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(serialize_cross_sensor_residual_fit(fit_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


@lru_cache(maxsize=8)
def load_cross_sensor_residual_fit(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))
