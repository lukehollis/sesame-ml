"""Load and validate the physical Sesame MuJoCo model."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from sesame_ml.constants import JOINT_NAMES, STAND_ANGLES_RAD

_ASSET_STL_NAMES = (
    "Internal-Frame-v121.stl",
    "Bottom-Cover-v121.stl",
    "Top-Cover-Enclosed-v117.stl",
    "L1-v117.stl",
    "L2-v117.stl",
    "L3-v117.stl",
    "L4-v117.stl",
    "R1-v117.stl",
    "R2-v117.stl",
    "R3-v117.stl",
    "R4-v117.stl",
)


def package_root() -> Path:
    return Path(__file__).resolve().parent


def mjcf_path() -> Path:
    return package_root() / "assets/mjcf/sesame.xml"


def mesh_directory() -> Path:
    """Return the package-owned printable mesh directory."""

    return package_root() / "assets/meshes"


def _asset_files() -> dict[str, bytes]:
    paths = {name: mesh_directory() / name for name in _ASSET_STL_NAMES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The sesame-ml installation is missing packaged mesh assets: "
            + ", ".join(missing)
        )
    return {name: path.read_bytes() for name, path in paths.items()}


def load_model() -> mujoco.MjModel:
    """Compile an independent MJCF model with the packaged printable meshes.

    Each environment receives its own ``MjModel`` because domain randomization mutates model
    parameters such as friction, mass, actuator gain, and force limits.
    """

    xml = mjcf_path().read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml, assets=_asset_files())
    validate_model(model)
    return model


def make_data(model: mujoco.MjModel, *, settle: bool = False) -> mujoco.MjData:
    """Create data in the firmware stand pose."""

    data = mujoco.MjData(model)
    set_stand_pose(model, data)
    mujoco.mj_forward(model, data)
    if settle:
        for _ in range(round(0.25 / model.opt.timestep)):
            data.ctrl[:] = STAND_ANGLES_RAD
            mujoco.mj_step(model, data)
    return data


def set_stand_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset the free base and all named joints to the calibrated stand pose."""

    mujoco.mj_resetData(model, data)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qpos = model.jnt_qposadr[root_id]
    data.qpos[root_qpos : root_qpos + 7] = np.asarray([0, 0, 0.049, 1, 0, 0, 0])
    for name, angle in zip(JOINT_NAMES, STAND_ANGLES_RAD, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    data.ctrl[:] = STAND_ANGLES_RAD


def joint_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in JOINT_NAMES
        ],
        dtype=np.int32,
    )


def joint_dof_addresses(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in JOINT_NAMES
        ],
        dtype=np.int32,
    )


def validate_model(model: mujoco.MjModel) -> None:
    """Fail early if joint/actuator conventions drift away from the physical robot."""

    if model.nu != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} actuators, model has {model.nu}")
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
    ]
    expected = [f"{name}_servo" for name in JOINT_NAMES]
    if actuator_names != expected:
        raise ValueError(
            f"Actuator order {actuator_names} does not match firmware order {expected}"
        )
    for name in JOINT_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            raise ValueError(f"Model is missing joint {name}")
