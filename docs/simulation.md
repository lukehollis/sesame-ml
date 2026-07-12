# Simulation, training and evaluation

## Model provenance

The package ships a MuJoCo MJCF and URDF built from the printable Sesame parts. It does not
need a separate CAD checkout.

- Visual geometry: packaged Sesame Robot Project STLs in millimetres, compiled at 0.001 scale.
- Kinematics: CAD/Onshape joint frames converted to `x-forward, y-left, z-up`.
- Joint order and stand pose: firmware order `[R1, R2, L1, L2, R4, R3, L3, L4]` and
  `[135, 45, 45, 135, 0, 180, 0, 180]` degrees.
- Contact: body envelope, upper/lower link capsules and rounded feet aligned to the printable
  surface. Detailed mesh triangles are render-only.
- Initial dynamics: approximately 384 g total, including a 280 g body/payload estimate.
- Actuation: saturated 0.18 Nm position servos with rate, lag and delay.

The starting mass, inertia, torque and camera/IMU transforms are plausible engineering values,
not measurements of every build. Weigh the finished rigid groups, measure servo response and
update the model/ranges before serious sim-to-real work.

## Physics and timing

| Loop | Default rate |
|---|---:|
| MuJoCo physics | 500 Hz |
| policy/control | 50 Hz |
| RGB capture/hold | 15 Hz |
| Orange Pi PWM | 50 Hz |

Native MuJoCo and MJX both filter desired targets through a randomized first-order response,
0-60 ms delay and a 600 degree/s maximum target slew. Native domain randomization changes
mass/CoM, friction, paired servo gain+bias, force, battery strength, delay, response and IMU
noise. Navigation additionally changes target and obstacle placement.

Nominal upright resets settle their initial contact transient before the episode begins.
Recovery resets are instead lifted clear of floor penetration and left in a deliberately
challenging orientation.

## Environments

### `SesameStand-v0`

Maintain the firmware stand pose with low angular motion, energy and slip. Success requires
surviving the full episode without a physical fall.

### `SesameRecovery-v0`

Starts at a large tilt or side fall outside the upright threshold. Success requires at least
0.5 seconds of stable height, orientation and angular rate; briefly crossing the threshold is
not sufficient.

### `SesameLocomotion-v0`

Tracks randomized forward/backward and yaw commands. Success requires the full episode,
upright mean at least 0.75, linear tracking RMSE no more than 0.06 m/s and yaw tracking RMSE no
more than 0.45 rad/s. A stationary policy is a failed locomotion policy even if it never falls.

### `SesameNavigation-v0`

Places a green goal and two collidable obstacles in a randomized scene. The environment's
reference command generator uses direct ground-truth goal bearing/distance; it does not invoke
the separate `OccupancyGridPlanner`, and therefore is a command baseline rather than an
obstacle-aware planner. The default actor still receives only the resulting velocity/yaw
command, not an exact goal vector. VLA adapters receive RGB, language and eight-joint state.
Success requires reaching within 0.12 m; leaving the 3 m arena is a distinct out-of-bounds
failure rather than a fall.

## Observations and privileged state

Feedback-free MG90S servos do not report angle. The deployable actor therefore receives the
last commanded target rather than perfect simulated qpos. Joint velocity, foot contacts and
goal-relative truth are zero unless the explicit privileged option is enabled. IMU inputs are
up vector, acceleration and gyro. See [Architecture](architecture.md) for exact slices.

This distinction is a training constraint: a policy that requires privileged state cannot be
silently deployed through the Orange Pi bridge.

## Evaluation artifacts

```bash
uv run sesame-ml evaluate \
  --env SesameStand-v0 \
  --env SesameRecovery-v0 \
  --env SesameLocomotion-v0 \
  --env SesameNavigation-v0 \
  --policy cpg \
  --episodes 20 \
  --videos 2 \
  --video-view split \
  --output artifacts/cpg-suite
```

Reports contain:

- return, success, fall and termination cause;
- upright/height statistics;
- command and linear/yaw tracking RMSE;
- path distance and initial/final/minimum goal distance;
- obstacle contact steps/fraction;
- energy and action rate/saturation;
- inference latency and execution real-time factor;
- all sampled domain parameters;
- the exact video path and replayable environment seed.

Video encoding time is excluded from execution throughput. MP4 is canonical because H.264
preserves fractional frame rates and terminal frames; GIF timing is approximate.

`--video-view external` records the task-aware observer camera, `front` records the exact held
policy RGB, and `split` adds that RGB as a picture-in-picture inset. Navigation observer video
frames the complete current robot-to-goal route and expands if a bad policy drifts.

## Native PPO

Stable-Baselines3 training supports state or pixel policies, vector environments, matched
training/evaluation normalization, periodic/best/final checkpoints, resume, TensorBoard and
interrupt artifacts. Use multiple seeds and both fixed/randomized evaluation. Do not choose a
checkpoint using return alone.

## Playground/MJX PPO

The optional Playground path uses a 20-value deployable actor and a 61-value asymmetric
critic. It supports batched mass/CoM/friction/servo randomization and JAX or Warp physics. Warp
requires an NVIDIA GPU; CPU JAX is suitable only for contract checks because compilation can
consume substantial memory.

```bash
uv sync --extra playground-cuda --extra dev
uv run sesame-ml train-mjx \
  --steps 100000000 --num-envs 4096 --impl warp \
  --output runs/mjx-locomotion
```

## Online planning

`sesame-ml plan` runs cross-entropy trajectory optimization from the live MuJoCo state and
executes only the first action chunk before replanning. The waypoint planner uses an occupancy
grid and inflates obstacles by the robot footprint. These are real planners, but their model
parameters still require physical calibration before hardware use.

## Behavior gates

Before calling a policy successful, verify:

1. strict task success is non-zero across held-out seeds;
2. no baseline that stands still passes locomotion;
3. randomized performance does not collapse;
4. actions respect rate/position limits without persistent saturation;
5. video agrees with metrics and exposes lateral drift/course context;
6. network dropout and delayed chunks enter fallback;
7. the physical trial begins raised/current-limited.
