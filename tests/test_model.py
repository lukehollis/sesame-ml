from __future__ import annotations

import mujoco
import numpy as np
import pytest

from sesame_ml.constants import JOINT_NAMES, STAND_ANGLES_RAD
from sesame_ml.model import joint_qpos_addresses, load_model, make_data


def test_model_compiles_with_firmware_action_order() -> None:
    model = load_model()
    assert model.nu == 8
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]
    assert names == [f"{name}_servo" for name in JOINT_NAMES]


def test_stand_pose_is_stable_for_one_second() -> None:
    model = load_model()
    data = make_data(model)
    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.allclose(data.qpos[joint_qpos_addresses(model)], STAND_ANGLES_RAD, atol=0.025)
    assert 0.045 < data.qpos[2] < 0.049
    assert abs(data.qpos[3]) > 0.98
    touch_force = 0.0
    for name in ("r3", "r4", "l3", "l4"):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{name}_touch")
        touch_force += float(data.sensordata[model.sensor_adr[sensor_id]])
    assert touch_force == pytest.approx(model.body_mass.sum() * 9.81, rel=0.03)
    assert all(data.contact[index].dim == 6 for index in range(data.ncon))


def test_model_includes_deployment_sensors() -> None:
    model = load_model()
    for name in ("imu_quat", "imu_gyro", "imu_accel", "imu_velocity"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera") >= 0
