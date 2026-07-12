from __future__ import annotations

import mujoco
import numpy as np
import pytest

from sesame_ml.constants import JOINT_NAMES, STAND_ANGLES_RAD
from sesame_ml.urdf import (
    assert_urdf_equivalent,
    forward_kinematics,
    urdf_path,
    validate_urdf_structure,
)


def test_urdf_structure_assets_inertias_and_limits() -> None:
    validate_urdf_structure()


def test_urdf_fk_matches_mujoco_at_stand_and_sampled_poses() -> None:
    report = assert_urdf_equivalent(samples=32, seed=20260712)
    assert report.poses_checked == 33
    assert report.frames_per_pose == 15
    assert report.comparisons == 495
    assert report.visual_meshes_checked == 11
    assert report.collision_groups_checked == 13
    assert report.maximum_position_error_m < 1e-9
    assert report.maximum_orientation_error_rad < 1e-7


def test_urdf_loads_with_mujoco_external_urdf_importer() -> None:
    imported = mujoco.MjModel.from_xml_path(str(urdf_path()))
    names = {
        mujoco.mj_id2name(imported, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(imported.njnt)
    }
    assert names == set(JOINT_NAMES)
    assert imported.njnt == 8
    assert imported.ngeom == 25
    # MuJoCo's URDF importer treats the root link as a fixed world body and therefore excludes
    # its 0.280 kg mass; all eight articulated link inertias survive the independent import.
    assert np.isclose(imported.body_mass.sum(), 0.104)


def test_urdf_stand_fk_contains_robot_and_sensor_frames() -> None:
    transforms = forward_kinematics(STAND_ANGLES_RAD)
    assert {
        "base",
        "r1_link",
        "r2_link",
        "l1_link",
        "l2_link",
        "r3_link",
        "r4_link",
        "l3_link",
        "l4_link",
        "r3_foot",
        "r4_foot",
        "l3_foot",
        "l4_foot",
        "imu",
        "front_camera",
    } == transforms.keys()
    for transform in transforms.values():
        assert transform.shape == (4, 4)
        np.testing.assert_allclose(transform[3], [0, 0, 0, 1])
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-9)


def test_urdf_fk_rejects_noncanonical_action_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        forward_kinematics(np.zeros(7))
