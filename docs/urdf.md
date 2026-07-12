# URDF model and validation

The production robot description is
[`src/sesame_ml/assets/urdf/sesame.urdf`](../src/sesame_ml/assets/urdf/sesame.urdf).
It is generated from the same corrected CAD measurements and dynamics parameters as the
MuJoCo model, but it is checked in as a readable, versioned artifact rather than generated at
install time.

Use the MJCF for Sesame's MuJoCo environments. Use the URDF when importing the robot into
Isaac Lab, PyBullet, RViz, Pinocchio or another URDF-based tool. Mesh filenames are relative to
the URDF (`../meshes/...`), so they resolve from a source checkout and from an installed wheel
without a ROS package index or a machine-specific absolute path.

Locate the installed model without assuming a virtual-environment layout:

```bash
python - <<'PY'
from sesame_ml.urdf import urdf_path
print(urdf_path())
PY
```

## Coordinate and command convention

- Robot frame: `+x` forward, `+y` left, `+z` up.
- Revolute axis: `0 0 -1` in every joint frame, matching the MJCF.
- Joint value: absolute physical servo angle in radians.
- Firmware order: `R1, R2, L1, L2, R4, R3, L3, L4`.
- Effort limit: `0.18 N m`.
- Velocity limit: `10.471975512 rad/s` (600 degrees/s).

MuJoCo represents each calibrated zero with a joint `ref`. URDF has no equivalent field, so
the reference rotation is incorporated into the URDF joint origin. Consequently, the same
absolute eight-element servo vector produces the same pose in both descriptions. Do not add a
second offset or reverse an axis in an importer.

| Leg | Kinematic chain | Hard servo limits |
| --- | --- | --- |
| front right | `base -> R1 -> r1_link -> R3 -> r3_link -> r3_foot` | R1 45–180°, R3 0–180° |
| rear right | `base -> R2 -> r2_link -> R4 -> r4_link -> r4_foot` | R2 0–135°, R4 0–180° |
| front left | `base -> L1 -> l1_link -> L3 -> l3_link -> l3_foot` | L1 0–135°, L3 0–180° |
| rear left | `base -> L2 -> l2_link -> L4 -> l4_link -> l4_foot` | L2 45–180°, L4 0–180° |

The distal MJCF hinges retain a five-degree numerical margin around the endpoints for contact
solver robustness, while the MJCF actuator and URDF hard limits are both 0–180 degrees. This
prevents importers that ignore ROS soft-limit extensions from commanding beyond a physical
servo endpoint.

## Geometry, dynamics and frames

The visual geometry uses all 11 printable STL parts at a `0.001` millimetre-to-metre scale.
Each visual mesh and local transform is checked directly against its MJCF counterpart.

Collision is intentionally simpler than the printable surface:

- one base box;
- each MuJoCo hip capsule represented by a cylinder and two spherical caps;
- each shin capsule represented by a cylinder, a proximal cap and the enclosing foot sphere;
- four 7 mm foot spheres.

This preserves the contact envelope while remaining portable across URDF engines that do not
support a capsule primitive. Cylinder centers, axes, lengths and radii are checked against the
MJCF `fromto` definitions.

The `imu` fixed link matches the MuJoCo IMU site. The `front_camera` fixed link matches the
MuJoCo camera coordinate frame (`+x` image-right, `+y` image-up, `-z` viewing direction). The
four named foot links match the four MuJoCo contact sites. URDF carries the camera frame but not
a camera sensor plugin; configure the MuJoCo model's 82-degree vertical field of view in the
target engine when policy pixels must match.

Masses and diagonal inertias are identical to the MJCF and total 0.384 kg. They remain
engineering estimates until measured on the assembled robot; replacing them requires updating
both descriptions and retaining the equivalence checks.

## Reproduce the validation

```bash
uv run python scripts/validate_urdf.py --samples 256 --seed 20260712
uv run pytest tests/test_urdf.py
```

The validator checks:

1. XML tree connectivity and unique link/joint names.
2. Exact firmware joint serialization, axes, hard limits, effort and inertial validity.
3. Portable mesh resolution, scale, part selection and visual transforms.
4. Base, link, foot, camera and IMU forward kinematics at stand and deterministic sampled
   servo poses.
5. Collision primitive dimensions and axes against the MJCF.
6. Independent import through MuJoCo's URDF parser.

The command exits nonzero if position error exceeds `1e-9 m` or orientation error exceeds
`1e-7 rad`. It prints the actual maxima and number of comparisons as JSON so CI logs retain
numerical evidence rather than only a pass/fail result.

The release validation covers the native MuJoCo importer and an independent
`yourdfpy`/`trimesh` load. PyBullet and Isaac Lab should still be imported and exercised on the
target Ubuntu/CUDA host before treating either engine as a qualified training backend; a native
PyBullet wheel was not available on the macOS ARM release machine.
