"""URDF access, structural validation, and MuJoCo equivalence checks.

The URDF deliberately uses physical servo radians as its revolute coordinates.  Joint
origins absorb MuJoCo's nonzero ``ref`` values, making a firmware action vector usable in
both representations without a second calibration convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import acos, cos, sin
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np

from sesame_ml.constants import JOINT_LIMITS_RAD, JOINT_NAMES, STAND_ANGLES_RAD
from sesame_ml.model import joint_qpos_addresses, load_model, make_data, mjcf_path, package_root

_EXPECTED_DYNAMIC_LINKS = (
    "base",
    "r1_link",
    "r2_link",
    "l1_link",
    "l2_link",
    "r4_link",
    "r3_link",
    "l3_link",
    "l4_link",
)
_FRAME_MAP = {
    "base": ("body", "base"),
    "r1_link": ("body", "r1_link"),
    "r2_link": ("body", "r2_link"),
    "l1_link": ("body", "l1_link"),
    "l2_link": ("body", "l2_link"),
    "r4_link": ("body", "r4_link"),
    "r3_link": ("body", "r3_link"),
    "l3_link": ("body", "l3_link"),
    "l4_link": ("body", "l4_link"),
    "r3_foot": ("site", "r3_contact"),
    "r4_foot": ("site", "r4_contact"),
    "l3_foot": ("site", "l3_contact"),
    "l4_foot": ("site", "l4_contact"),
    "imu": ("site", "imu"),
    "front_camera": ("camera", "front_camera"),
}
_VISUAL_MAP = {
    "internal_frame_visual": "internal_frame_visual",
    "bottom_cover_visual": "bottom_cover_visual",
    "top_cover_visual": "top_cover_visual",
    "r1_visual": "r1_visual",
    "r2_visual": "r2_visual",
    "l1_visual": "l1_visual",
    "l2_visual": "l2_visual",
    "r4_visual": "r4_visual",
    "r3_visual": "r3_visual",
    "l3_visual": "l3_visual",
    "l4_visual": "l4_visual",
}
_CAPSULE_COLLISION_MAP = {
    "r1_collision_cylinder": "r1_collision",
    "r2_collision_cylinder": "r2_collision",
    "l1_collision_cylinder": "l1_collision",
    "l2_collision_cylinder": "l2_collision",
    "r4_shin_cylinder": "r4_shin",
    "r3_shin_cylinder": "r3_shin",
    "l3_shin_cylinder": "l3_shin",
    "l4_shin_cylinder": "l4_shin",
}
_FOOT_COLLISION_MAP = {
    "r3_foot_collision": "r3_foot",
    "r4_foot_collision": "r4_foot",
    "l3_foot_collision": "l3_foot",
    "l4_foot_collision": "l4_foot",
}


@dataclass(frozen=True)
class UrdfValidationReport:
    """Numerical evidence from a MuJoCo-to-URDF kinematic comparison."""

    poses_checked: int
    frames_per_pose: int
    comparisons: int
    visual_meshes_checked: int
    collision_groups_checked: int
    maximum_position_error_m: float
    maximum_orientation_error_rad: float
    seed: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def urdf_path() -> Path:
    """Return the installed production URDF path."""

    return package_root() / "assets/urdf/sesame.urdf"


def resolve_mesh_uri(uri: str) -> Path:
    """Resolve a URDF-relative mesh URI in source and wheel installs."""

    if Path(uri).is_absolute() or uri.startswith("package://"):
        raise ValueError(f"URDF mesh URI must be portable and relative, got {uri!r}")
    candidate = (urdf_path().parent / uri).resolve()
    root = package_root().resolve()
    if root not in candidate.parents:
        raise ValueError(f"Mesh URI escapes sesame_ml: {uri!r}")
    return candidate


def _vector(text: str | None, *, length: int = 3) -> np.ndarray:
    if text is None:
        return np.zeros(length, dtype=np.float64)
    value = np.fromstring(text, sep=" ", dtype=np.float64)
    if value.shape != (length,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Expected {length} finite values, got {text!r}")
    return value


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0:
        raise ValueError("Quaternion must be nonzero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("Revolute joint axis must be nonzero")
    x, y, z = axis / norm
    c, s, one_minus_c = cos(angle), sin(angle), 1.0 - cos(angle)
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def _transform(origin: ElementTree.Element | None) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    if origin is not None:
        value[:3, 3] = _vector(origin.get("xyz"))
        value[:3, :3] = _rpy_matrix(_vector(origin.get("rpy")))
    return value


def _joint_transform(joint: ElementTree.Element, position: float) -> np.ndarray:
    value = _transform(joint.find("origin"))
    if joint.get("type") in {"revolute", "continuous"}:
        axis_element = joint.find("axis")
        axis = _vector(axis_element.get("xyz") if axis_element is not None else "1 0 0")
        rotation = np.eye(4, dtype=np.float64)
        rotation[:3, :3] = _axis_angle_matrix(axis, position)
        value = value @ rotation
    elif joint.get("type") == "prismatic":
        axis_element = joint.find("axis")
        axis = _vector(axis_element.get("xyz") if axis_element is not None else "1 0 0")
        translation = np.eye(4, dtype=np.float64)
        translation[:3, 3] = axis / np.linalg.norm(axis) * position
        value = value @ translation
    elif joint.get("type") != "fixed":
        raise ValueError(f"Unsupported URDF joint type {joint.get('type')!r}")
    return value


def forward_kinematics(
    joint_positions: dict[str, float] | np.ndarray | list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute base-relative link transforms from the packaged URDF.

    Arrays use canonical firmware order.  Mapping values may omit fixed joints but must
    contain every actuated joint.
    """

    if joint_positions is None:
        positions = dict(zip(JOINT_NAMES, STAND_ANGLES_RAD, strict=True))
    elif isinstance(joint_positions, dict):
        positions = {name: float(joint_positions[name]) for name in JOINT_NAMES}
    else:
        vector = np.asarray(joint_positions, dtype=np.float64)
        if vector.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(vector)):
            raise ValueError(f"joint_positions must have shape ({len(JOINT_NAMES)},)")
        positions = dict(zip(JOINT_NAMES, vector, strict=True))

    root = ElementTree.parse(urdf_path()).getroot()
    child_joints: dict[str, list[ElementTree.Element]] = {}
    child_links = set()
    for joint in root.findall("joint"):
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        if parent_element is None or child_element is None:
            raise ValueError(f"Joint {joint.get('name')} is missing a parent or child")
        parent, child = parent_element.get("link"), child_element.get("link")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint.get('name')} has an empty parent or child")
        child_joints.setdefault(parent, []).append(joint)
        if child in child_links:
            raise ValueError(f"Link {child!r} has multiple parents")
        child_links.add(child)

    links = {element.get("name") for element in root.findall("link")}
    roots = links - child_links
    if roots != {"base"}:
        raise ValueError(f"Expected base as the sole URDF root, found {sorted(roots)}")

    result = {"base": np.eye(4, dtype=np.float64)}
    pending = ["base"]
    while pending:
        parent = pending.pop()
        for joint in child_joints.get(parent, []):
            child_element = joint.find("child")
            assert child_element is not None
            child = child_element.get("link")
            assert child is not None
            position = positions.get(joint.get("name", ""), 0.0)
            result[child] = result[parent] @ _joint_transform(joint, position)
            pending.append(child)
    if result.keys() != links:
        raise ValueError(
            f"URDF is disconnected; unreachable links: {sorted(links - result.keys())}"
        )
    return result


def validate_urdf_structure() -> None:
    """Validate the URDF tree, assets, firmware ordering, limits, and inertias."""

    root = ElementTree.parse(urdf_path()).getroot()
    if root.tag != "robot" or root.get("name") != "sesame_quadruped":
        raise ValueError("URDF root must be robot[name='sesame_quadruped']")

    link_elements = root.findall("link")
    link_names = [element.get("name") for element in link_elements]
    if len(link_names) != len(set(link_names)) or None in link_names:
        raise ValueError("URDF link names must be present and unique")

    joint_elements = root.findall("joint")
    joint_names = [element.get("name") for element in joint_elements]
    if len(joint_names) != len(set(joint_names)) or None in joint_names:
        raise ValueError("URDF joint names must be present and unique")
    actuated = tuple(
        str(element.get("name")) for element in joint_elements if element.get("type") != "fixed"
    )
    if actuated != JOINT_NAMES:
        raise ValueError(f"URDF order {actuated} does not match firmware order {JOINT_NAMES}")

    for index, name in enumerate(JOINT_NAMES):
        joint = root.find(f"joint[@name='{name}']")
        assert joint is not None
        axis = joint.find("axis")
        if axis is None or not np.allclose(_vector(axis.get("xyz")), [0, 0, -1]):
            raise ValueError(f"Joint {name} does not preserve the MuJoCo servo axis")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Joint {name} has no limit")
        lower, upper = float(limit.get("lower", "nan")), float(limit.get("upper", "nan"))
        expected = JOINT_LIMITS_RAD[index]
        if not np.allclose([lower, upper], expected, atol=1e-9):
            raise ValueError(f"Joint {name} limits drifted from the firmware")
        if float(limit.get("effort", "nan")) != 0.18:
            raise ValueError(f"Joint {name} does not preserve the actuator effort limit")

    mjcf = ElementTree.parse(mjcf_path()).getroot()
    total_mass = 0.0
    for name in _EXPECTED_DYNAMIC_LINKS:
        link = root.find(f"link[@name='{name}']")
        assert link is not None
        inertial, mass, inertia = (
            link.find("inertial"),
            link.find("inertial/mass"),
            link.find("inertial/inertia"),
        )
        if inertial is None or mass is None or inertia is None:
            raise ValueError(f"Dynamic link {name} is missing inertial data")
        link_mass = float(mass.get("value", "nan"))
        total_mass += link_mass
        tensor = np.asarray(
            [
                [
                    float(inertia.get("ixx", "nan")),
                    float(inertia.get("ixy", "nan")),
                    float(inertia.get("ixz", "nan")),
                ],
                [
                    float(inertia.get("ixy", "nan")),
                    float(inertia.get("iyy", "nan")),
                    float(inertia.get("iyz", "nan")),
                ],
                [
                    float(inertia.get("ixz", "nan")),
                    float(inertia.get("iyz", "nan")),
                    float(inertia.get("izz", "nan")),
                ],
            ]
        )
        if not np.all(np.linalg.eigvalsh(tensor) > 0):
            raise ValueError(f"Dynamic link {name} has a non-positive inertia tensor")
        mjcf_inertial = mjcf.find(f".//body[@name='{name}']/inertial")
        if mjcf_inertial is None:
            raise ValueError(f"MJCF dynamic link {name} is missing inertial data")
        urdf_inertial_origin = _transform(inertial.find("origin"))
        if not np.allclose(urdf_inertial_origin[:3, 3], _vector(mjcf_inertial.get("pos"))):
            raise ValueError(f"Inertial origin {name} drifted from the MJCF")
        if not np.isclose(link_mass, float(mjcf_inertial.get("mass", "nan"))):
            raise ValueError(f"Mass {name} drifted from the MJCF")
        if not np.allclose(np.diag(tensor), _vector(mjcf_inertial.get("diaginertia"))):
            raise ValueError(f"Inertia {name} drifted from the MJCF")
    if not np.isclose(total_mass, 0.384, atol=1e-12):
        raise ValueError(f"URDF mass is {total_mass}, expected 0.384 kg")

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None or not resolve_mesh_uri(filename).is_file():
            raise FileNotFoundError(f"Missing URDF mesh {filename!r}")
        if not np.allclose(_vector(mesh.get("scale")), [0.001, 0.001, 0.001]):
            raise ValueError(f"Mesh {filename!r} must preserve millimetre-to-metre scale")

    mjcf_meshes = {
        str(element.get("name")): str(element.get("file")) for element in mjcf.findall("asset/mesh")
    }
    for visual_name in _VISUAL_MAP:
        visual = root.find(f".//visual[@name='{visual_name}']")
        mjcf_geom = mjcf.find(f".//geom[@name='{visual_name}']")
        if visual is None or mjcf_geom is None:
            raise ValueError(f"Visual {visual_name} must exist in both robot descriptions")
        urdf_local = _transform(visual.find("origin"))
        mjcf_local = np.eye(4, dtype=np.float64)
        mjcf_local[:3, 3] = _vector(mjcf_geom.get("pos"))
        mjcf_local[:3, :3] = _quaternion_matrix(_vector(mjcf_geom.get("quat", "1 0 0 0"), length=4))
        if not np.allclose(urdf_local, mjcf_local, atol=2e-9):
            raise ValueError(f"Visual transform {visual_name} drifted from the MJCF")
        urdf_mesh = visual.find("geometry/mesh")
        mesh_name = mjcf_geom.get("mesh")
        if urdf_mesh is None or mesh_name is None:
            raise ValueError(f"Visual {visual_name} is missing a mesh")
        if Path(str(urdf_mesh.get("filename"))).name != Path(mjcf_meshes[mesh_name]).name:
            raise ValueError(f"Visual mesh {visual_name} drifted from the MJCF")

    mjcf_body_collision = mjcf.find(".//geom[@name='body_collision']")
    urdf_body_collision = root.find(".//collision[@name='body_collision']")
    if mjcf_body_collision is None or urdf_body_collision is None:
        raise ValueError("Both descriptions must contain body_collision")
    urdf_body_transform = _transform(urdf_body_collision.find("origin"))
    if not np.allclose(urdf_body_transform[:3, 3], _vector(mjcf_body_collision.get("pos"))):
        raise ValueError("Body collision origin drifted from the MJCF")
    urdf_body_size = _vector(urdf_body_collision.find("geometry/box").get("size"))
    if not np.allclose(urdf_body_size, 2 * _vector(mjcf_body_collision.get("size"))):
        raise ValueError("Body collision size drifted from the MJCF")

    for collision_name, mjcf_name in _CAPSULE_COLLISION_MAP.items():
        collision = root.find(f".//collision[@name='{collision_name}']")
        mjcf_geom = mjcf.find(f".//geom[@name='{mjcf_name}']")
        if collision is None or mjcf_geom is None:
            raise ValueError(f"Collision approximation {collision_name} is missing")
        cylinder = collision.find("geometry/cylinder")
        if cylinder is None:
            raise ValueError(f"Collision approximation {collision_name} must be a cylinder")
        endpoints = _vector(mjcf_geom.get("fromto"), length=6).reshape(2, 3)
        segment = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(segment))
        local = _transform(collision.find("origin"))
        if not np.allclose(local[:3, 3], endpoints.mean(axis=0), atol=1e-10):
            raise ValueError(f"Collision center {collision_name} drifted from the MJCF")
        if not np.allclose(local[:3, 2], segment / length, atol=1e-10):
            raise ValueError(f"Collision axis {collision_name} drifted from the MJCF")
        if not np.isclose(float(cylinder.get("length", "nan")), length, atol=1e-12):
            raise ValueError(f"Collision length {collision_name} drifted from the MJCF")
        radius = float(_vector(mjcf_geom.get("size"), length=1)[0])
        if not np.isclose(float(cylinder.get("radius", "nan")), radius, atol=1e-12):
            raise ValueError(f"Collision radius {collision_name} drifted from the MJCF")
        prefix = collision_name.removesuffix("_cylinder")
        cap_names = [f"{prefix}_proximal_cap"]
        cap_positions = [endpoints[0]]
        if "_collision_cylinder" in collision_name:
            cap_names.append(f"{prefix}_distal_cap")
            cap_positions.append(endpoints[1])
        for cap_name, expected_position in zip(cap_names, cap_positions, strict=True):
            cap = root.find(f".//collision[@name='{cap_name}']")
            if cap is None:
                raise ValueError(f"Capsule approximation {cap_name} is missing")
            sphere = cap.find("geometry/sphere")
            if sphere is None or not np.isclose(
                float(sphere.get("radius", "nan")), radius, atol=1e-12
            ):
                raise ValueError(f"Capsule radius {cap_name} drifted from the MJCF")
            if not np.allclose(
                _transform(cap.find("origin"))[:3, 3], expected_position, atol=1e-12
            ):
                raise ValueError(f"Capsule origin {cap_name} drifted from the MJCF")

    for collision_name, mjcf_name in _FOOT_COLLISION_MAP.items():
        collision = root.find(f".//collision[@name='{collision_name}']")
        mjcf_geom = mjcf.find(f".//geom[@name='{mjcf_name}']")
        if collision is None or mjcf_geom is None:
            raise ValueError(f"Foot collision {collision_name} is missing")
        sphere = collision.find("geometry/sphere")
        if sphere is None:
            raise ValueError(f"Foot collision {collision_name} must be a sphere")
        size_text = mjcf_geom.get("size")
        if size_text is None:
            foot_default = mjcf.find("default/default[@class='foot']/geom")
            if foot_default is None:
                raise ValueError("MJCF foot collision default is missing")
            size_text = foot_default.get("size")
        radius = float(_vector(size_text, length=1)[0])
        if not np.isclose(float(sphere.get("radius", "nan")), radius, atol=1e-12):
            raise ValueError(f"Foot radius {collision_name} drifted from the MJCF")
        if not np.allclose(
            _transform(collision.find("origin"))[:3, 3], _vector(mjcf_geom.get("pos"))
        ):
            raise ValueError(f"Foot origin {collision_name} drifted from the MJCF")

    # Also proves the graph is connected, acyclic, and has a single base root.
    forward_kinematics()


def _mujoco_transform(
    model: mujoco.MjModel, data: mujoco.MjData, kind: str, name: str
) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    if kind == "body":
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        value[:3, 3], value[:3, :3] = data.xpos[object_id], data.xmat[object_id].reshape(3, 3)
    elif kind == "site":
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        value[:3, 3], value[:3, :3] = (
            data.site_xpos[object_id],
            data.site_xmat[object_id].reshape(3, 3),
        )
    elif kind == "camera":
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        value[:3, 3], value[:3, :3] = (
            data.cam_xpos[object_id],
            data.cam_xmat[object_id].reshape(3, 3),
        )
    else:  # pragma: no cover - internal constant protects this branch
        raise ValueError(f"Unknown MuJoCo frame kind {kind!r}")
    return value


def _pose_error(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    position_error = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
    delta = actual[:3, :3].T @ expected[:3, :3]
    orientation_error = acos(float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)))
    return position_error, orientation_error


def compare_urdf_to_mjcf(*, samples: int = 64, seed: int = 20260712) -> UrdfValidationReport:
    """Compare URDF and MuJoCo FK at stand and deterministic random servo poses."""

    if samples < 0:
        raise ValueError("samples must be nonnegative")
    validate_urdf_structure()
    model = load_model()
    data = make_data(model)
    qpos_addresses = joint_qpos_addresses(model)
    random = np.random.default_rng(seed)
    poses = [STAND_ANGLES_RAD]
    poses.extend(
        random.uniform(JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1]) for _ in range(samples)
    )

    maximum_position_error = 0.0
    maximum_orientation_error = 0.0
    for pose in poses:
        data.qpos[qpos_addresses] = pose
        mujoco.mj_forward(model, data)
        urdf_frames = forward_kinematics(pose)
        base_world = _mujoco_transform(model, data, "body", "base")
        world_to_base = np.linalg.inv(base_world)
        for urdf_name, (kind, mjcf_name) in _FRAME_MAP.items():
            actual = world_to_base @ _mujoco_transform(model, data, kind, mjcf_name)
            expected = urdf_frames[urdf_name]
            position_error, orientation_error = _pose_error(actual, expected)
            maximum_position_error = max(maximum_position_error, position_error)
            maximum_orientation_error = max(maximum_orientation_error, orientation_error)

    return UrdfValidationReport(
        poses_checked=len(poses),
        frames_per_pose=len(_FRAME_MAP),
        comparisons=len(poses) * len(_FRAME_MAP),
        visual_meshes_checked=len(_VISUAL_MAP),
        collision_groups_checked=1 + len(_CAPSULE_COLLISION_MAP) + len(_FOOT_COLLISION_MAP),
        maximum_position_error_m=maximum_position_error,
        maximum_orientation_error_rad=maximum_orientation_error,
        seed=seed,
    )


def assert_urdf_equivalent(
    *,
    samples: int = 64,
    seed: int = 20260712,
    position_tolerance_m: float = 1e-9,
    orientation_tolerance_rad: float = 1e-7,
) -> UrdfValidationReport:
    """Raise when the production URDF diverges numerically from the MuJoCo model."""

    report = compare_urdf_to_mjcf(samples=samples, seed=seed)
    if report.maximum_position_error_m > position_tolerance_m:
        raise ValueError(
            f"URDF position error {report.maximum_position_error_m:.3e} m exceeds "
            f"{position_tolerance_m:.3e} m"
        )
    if report.maximum_orientation_error_rad > orientation_tolerance_rad:
        raise ValueError(
            f"URDF orientation error {report.maximum_orientation_error_rad:.3e} rad exceeds "
            f"{orientation_tolerance_rad:.3e} rad"
        )
    return report
