# Architecture and deployment contract

Sesame ML separates model inference from deterministic robot safety. The same policy boundary
can be exercised against a local MuJoCo process, over a loopback socket, across Wi-Fi to a
simulated robot, or across Wi-Fi to the physical Orange Pi runtime.

## Layers

```text
RGB / language / task goal
            |
  VLA, world model, route planner, or skill selector       policy machine
            |
  short action chunk, skill, or velocity command
            |
  learned policy / CEM / stabilizing controller            policy machine
            |
  timestamped N x 8 absolute joint targets
            |
  binary MessagePack over WebSocket
            |
  sequence/TTL checks -> latest-only buffer -> 50 Hz interpolation
            |
  calibrated limits -> slew limit -> watchdog              Orange Pi
            |
  PCA9685 PWM -> eight servos
```

A VLA may emit joint chunks directly, but the preferred complex-task architecture is
hierarchical: a slower vision/language model selects a goal or short skill while a 50 Hz
controller handles balance and locomotion. Direct chunks still pass through exactly the same
limits, freshness checks and watchdog.

## Frames and joints

- Robot frame: `+x` forward toward the face/camera, `+y` left, `+z` up.
- Quaternion convention on the wire: scalar-first `(w, x, y, z)`.
- Firmware, dataset, wire and actuator order: `[R1, R2, L1, L2, R4, R3, L3, L4]`.
- Remote actions: absolute joint targets in radians.
- Gym/MJX actions: eight normalized residuals around the stand pose.
- Control period: 20 ms (50 Hz).

Never send a normalized residual directly to the absolute-radian endpoint. Typed policy and
transport objects reject wrong shapes, non-finite values and targets outside the global
physical servo range. The Orange Pi then applies the narrower, measured per-robot YAML limits;
the host-side policy object does not know that calibration.

## Deployable observation boundary

The native Gym state has 43 values so state and pixel policies share one stable schema:

| Slice | Meaning | Physical source |
|---|---|---|
| `0:3` | body up vector | IMU quaternion |
| `3:6` | body accelerometer divided by 9.81 | IMU acceleration |
| `6:9` | body gyro, rad/s | IMU gyro |
| `9:17` | commanded joint offset from stand, rad | last servo target |
| `17:25` | joint velocity/10 or zero | zero without feedback servos |
| `25:33` | previous normalized action | controller memory |
| `33:36` | normalized velocity/yaw command | policy host/task source |
| `36:40` | contact truth or zero | zero by default |
| `40:43` | goal-relative truth or zero | zero by default |

Perfect simulated joints, contacts and goal-relative state appear only when the explicit
privileged flag is enabled. They are never silently exposed to a deployable actor. Pixel
observations add the front RGB frame. Physics remains at 500 Hz and control at 50 Hz while
camera frames are captured/held at 15 Hz by default, matching the remote deployment cadence.

The MJX actor uses the deployable 20-value subset: up, acceleration, gyro, command, and
commanded joint offset. Its critic additionally receives simulator state, forces and contacts.

The VLA contract is intentionally smaller and framework-neutral:

- absolute eight-joint commanded state;
- one RGB front image;
- natural-language instruction;
- capture timestamp and monotonic sequence;
- optional backend-specific context.

OpenPI and GR00T therefore cannot exploit Gym-only goal or contact truth.

## Action chunks and freshness

`ActionChunkV1` contains a monotonically increasing chunk ID, the observation sequence used
to produce it, a control period, an `N x 8` target array and a validity deadline. The robot:

1. accepts only a chunk based on an observation it actually sent;
2. rejects replayed/out-of-order chunks;
3. atomically replaces the previous chunk rather than draining a stale FIFO;
4. interpolates at 50 Hz using its local monotonic clock;
5. enters fallback when the chunk or connection becomes stale;
6. requests PCA9685 full-off in one transaction on timeout or orderly shutdown.

That last action is a software boundary, not a hang-independent fail-safe. A build that must
remove motion during an SBC, kernel, control-loop or I2C hang needs an independent watchdog on
PCA9685 `OE` or a rated normally-disabled servo-power switch.

Because the host echoes an observation sequence, host and robot clocks do not require
synchronization.

## Physics and actuator boundary

The MJCF and URDF ship inside the package. Rendering uses the printable meshes; collision uses
a measured body envelope, link capsules and rounded feet. Foot contact is aligned to the
printable surface and uses six-dimensional contact with bounded torsional/rolling friction.

Both native MuJoCo and MJX apply:

- 25-80 ms first-order servo target lag;
- 0-60 ms action delay;
- 600 degree/s target slew limit;
- calibrated joint limits;
- conservative 0.18 Nm force saturation;
- mass/CoM, friction, gain, torque, battery and IMU variation where supported.

Position-actuator gain and bias are randomized together so stiffness variation does not move
the commanded equilibrium.

## Data boundary

The canonical episode format is lossless, chunked, checksummed and crash-tolerant. Schema v2
stores raw RGB, language, absolute state/action radians, timing, pre-action IMU/command,
reward/termination, next-step task labels and episode domain/seed metadata. LeRobot, OpenPI and
GR00T datasets are derived artifacts; the canonical episode remains the source of truth.

## Deployment gates

A candidate checkpoint should pass, in order:

1. offline action shape/range and held-out prediction checks;
2. fixed-seed closed-loop simulation;
3. domain-randomized simulation;
4. network-path latency/dropout/watchdog evaluation;
5. raised, current-limited hardware checks;
6. tethered floor trials with an accessible cutoff;
7. only then, untethered task trials.

Report success, fall rate, upright time, tracking error, goal distance, obstacle contact,
energy/action rate, inference latency, stale chunks and watchdog entries. Simulator return or
VLA loss alone is never a deployment criterion.
