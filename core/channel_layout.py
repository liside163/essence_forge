from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


LEGACY_CORE_CHANNEL_NAMES: tuple[str, ...] = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "q0",
    "q1",
    "q2",
    "q3",
)

APPENDED_RAW_CHANNEL_NAMES: tuple[str, ...] = (
    "actuator_ctrl_0",
    "actuator_ctrl_1",
    "actuator_ctrl_2",
    "actuator_ctrl_3",
    "motor_rpm_1",
    "motor_rpm_2",
    "motor_rpm_3",
    "motor_rpm_4",
    "baro_alt",
    "baro_temp",
    "baro_pressure",
)

DEFAULT_30_RAW_CHANNEL_NAMES: tuple[str, ...] = (
    LEGACY_CORE_CHANNEL_NAMES + APPENDED_RAW_CHANNEL_NAMES
)

CROSS_SENSOR_RESIDUAL_CHANNEL_NAMES: tuple[str, ...] = (
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

LEGACY_CORE_CHANNEL_COUNT = len(LEGACY_CORE_CHANNEL_NAMES)
CROSS_SENSOR_RESIDUAL_CHANNEL_COUNT = len(CROSS_SENSOR_RESIDUAL_CHANNEL_NAMES)

RAW_SENSOR_GROUP_TEMPLATES: dict[str, tuple[str, ...]] = {
    "accel": ("accel_x", "accel_y", "accel_z"),
    "gyro": ("gyro_x", "gyro_y", "gyro_z"),
    "mag": ("mag_x", "mag_y", "mag_z"),
    "pos": ("pos_x", "pos_y", "pos_z"),
    "vel": ("vel_x", "vel_y", "vel_z"),
    "quat": ("q0", "q1", "q2", "q3"),
    "actuator_ctrl": (
        "actuator_ctrl_0",
        "actuator_ctrl_1",
        "actuator_ctrl_2",
        "actuator_ctrl_3",
    ),
    "motor_rpm": (
        "motor_rpm_1",
        "motor_rpm_2",
        "motor_rpm_3",
        "motor_rpm_4",
    ),
    "actuator_rpm": (
        "actuator_ctrl_0",
        "actuator_ctrl_1",
        "actuator_ctrl_2",
        "actuator_ctrl_3",
        "motor_rpm_1",
        "motor_rpm_2",
        "motor_rpm_3",
        "motor_rpm_4",
    ),
    "baro": ("baro_alt", "baro_temp", "baro_pressure"),
}

RESIDUAL_SENSOR_GROUP_TEMPLATES: dict[str, tuple[str, ...]] = {
    "res_imu_ekf": (
        "imu_ekf_vel_res_x",
        "imu_ekf_vel_res_y",
        "imu_ekf_vel_res_z",
    ),
    "res_pos_vel": ("pos_vel_res_x", "pos_vel_res_y", "pos_vel_res_z"),
    "res_accel_pos": (
        "accel_pos_res_x",
        "accel_pos_res_y",
        "accel_pos_res_z",
    ),
    "res_rpm_modal": (
        "rpm_modal_common_res",
        "rpm_modal_a_res",
        "rpm_modal_b_res",
        "rpm_modal_c_res",
    ),
    "res_baro": ("baro_pos_res", "baro_vel_res"),
    "res_qgyro": ("qgyro_x_res", "qgyro_y_res"),
}


def channel_names_from_specs(channels: Sequence[object] | Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    for channel in channels:
        if isinstance(channel, str):
            names.append(channel)
            continue
        name = getattr(channel, "name", None)
        if name is None:
            raise TypeError(f"Unsupported channel spec without .name: {channel!r}")
        names.append(str(name))
    return tuple(names)


def validate_unique_channel_names(channel_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in channel_names)
    duplicates = tuple(
        name for name, count in Counter(names).items() if int(count) > 1
    )
    if duplicates:
        raise ValueError(
            f"Channel names must be unique; duplicates={list(duplicates)}"
        )
    return names


def validate_legacy_core_prefix(channel_names: Sequence[str]) -> tuple[str, ...]:
    names = validate_unique_channel_names(channel_names)
    if len(names) < LEGACY_CORE_CHANNEL_COUNT:
        raise ValueError(
            "Channel layout must include the first 19 legacy core channels."
        )
    actual_prefix = names[:LEGACY_CORE_CHANNEL_COUNT]
    if actual_prefix != LEGACY_CORE_CHANNEL_NAMES:
        raise ValueError(
            "Channel layout first 19 channels must match the legacy core order; "
            f"expected={LEGACY_CORE_CHANNEL_NAMES}, actual={actual_prefix}"
        )
    return names


def _require_group_indices(
    channel_names: Sequence[str],
    group_members: Sequence[str],
) -> tuple[int, ...] | None:
    names = tuple(str(name) for name in channel_names)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    if not all(member in name_to_idx for member in group_members):
        return None
    return tuple(name_to_idx[member] for member in group_members)


def _build_sensor_groups(
    channel_names: Sequence[str],
    templates: Mapping[str, Sequence[str]],
) -> dict[str, tuple[int, ...]]:
    names = validate_unique_channel_names(channel_names)
    sensor_groups: dict[str, tuple[int, ...]] = {}
    for group_name, group_members in templates.items():
        indices = _require_group_indices(names, group_members)
        if indices is not None:
            sensor_groups[group_name] = indices
    return sensor_groups


def build_raw_sensor_groups(channel_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    return _build_sensor_groups(channel_names, RAW_SENSOR_GROUP_TEMPLATES)


def build_named_raw_groups(
    channel_names: Sequence[str],
    group_names: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    groups = build_raw_sensor_groups(channel_names)
    resolved: list[tuple[int, ...]] = []
    for group_name in group_names:
        normalized = str(group_name).strip()
        if normalized not in groups:
            raise ValueError(
                f"Unknown raw sensor group '{group_name}'. "
                f"Available groups: {sorted(groups.keys())}"
            )
        resolved.append(groups[normalized])
    return tuple(resolved)


def build_raw_plus_residual_sensor_groups(
    channel_names: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    sensor_groups = build_raw_sensor_groups(channel_names)
    sensor_groups.update(
        _build_sensor_groups(channel_names, RESIDUAL_SENSOR_GROUP_TEMPLATES)
    )
    return sensor_groups


def get_quaternion_indices(channel_names: Sequence[str]) -> tuple[int, int, int, int]:
    indices = _require_group_indices(channel_names, RAW_SENSOR_GROUP_TEMPLATES["quat"])
    if indices is None:
        raise ValueError("Quaternion channels q0/q1/q2/q3 are required in channel layout.")
    return indices  # type: ignore[return-value]


LEGACY_ACCEL_INDICES = build_raw_sensor_groups(LEGACY_CORE_CHANNEL_NAMES)["accel"]
LEGACY_GYRO_INDICES = build_raw_sensor_groups(LEGACY_CORE_CHANNEL_NAMES)["gyro"]
LEGACY_MAG_INDICES = build_raw_sensor_groups(LEGACY_CORE_CHANNEL_NAMES)["mag"]
LEGACY_POS_INDICES = build_raw_sensor_groups(LEGACY_CORE_CHANNEL_NAMES)["pos"]
LEGACY_VEL_INDICES = build_raw_sensor_groups(LEGACY_CORE_CHANNEL_NAMES)["vel"]
LEGACY_QUATERNION_INDICES = get_quaternion_indices(LEGACY_CORE_CHANNEL_NAMES)


def build_input_feature_names(
    raw_channel_names: Sequence[str],
    residual_channel_names: Sequence[str] = (),
    include_health_mask: bool = False,
) -> tuple[str, ...]:
    names = list(str(name) for name in raw_channel_names)
    names.extend(str(name) for name in residual_channel_names)
    if include_health_mask:
        names.extend(f"{name}_mask" for name in raw_channel_names)
    return tuple(names)
