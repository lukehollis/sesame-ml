# Safety and sim-to-real checklist

Sesame is small, but eight geared servos, lithium batteries and an autonomous network policy
can still pinch, overheat, stall, fall, damage wiring or corrupt the compute rail. Simulation
success is not permission to run an untethered robot.

## Non-negotiable boundaries

- The Orange Pi never powers servos through its USB/header rail.
- The PCA9685 logic `VCC` is not servo power `V+`.
- Every physical robot uses a measured calibration file; generic PWM endpoints are not a
  deployable calibration.
- Every command is range- and slew-limited locally after it crosses the network.
- Stale, replayed, malformed or expired chunks are rejected.
- Timeout, disconnect and orderly client exit request PWM full-off unless an explicitly
  validated local stand fallback was selected. This software path does not cover an SBC,
  kernel, control-loop or I2C hang.
- A DC-rated physical servo cutoff and correctly selected battery fuses remain accessible during
  development. Use an independently driven PCA9685 `OE` or rated normally-disabled power switch
  when automatic hang response is required.
- The public CLI WebSocket server is a trusted-LAN endpoint, not an internet control service.

## Calibration

Raise the robot and current-limit the servo supply. For one channel at a time, record:

1. firmware/PCA channel identity;
2. direction around a printed alignment pose;
3. subtrim;
4. minimum/maximum pulse verified on that servo;
5. mechanical angle endpoints with margin;
6. maximum safe command step/rate;
7. unloaded and loaded current/response.

The canonical channel order is `[R1, R2, L1, L2, R4, R3, L3, L4]`. Never infer mirrored
directions from a photo. Nominal servo ranges contain poses that may self-collide; robot-specific
limits must be narrower where required.

## Power checkout

Follow [the two-battery reference build](hardware-reference.md). Before connecting the robot:

- verify polarity and output voltage at every connector;
- fuse both batteries close to the positive terminal;
- verify compute 5 V and servo `V+` positives are not connected;
- return high servo current directly through the rated power distribution/regulator negative,
  and join the thin logic signal reference there;
- use wire/connectors/regulators rated from measured sustained and simultaneous transient
  current;
- test battery cutoff, voltage sag and regulator temperature;
- charge with chemistry/cell-count-correct chargers while disconnected and disarmed.

## Policy qualification

For each checkpoint/model revision:

1. Record exact weights/config, normalization and model/URDF revisions.
2. Run held-out offline action checks.
3. Run at least 50 fixed-seed and 50 randomized closed-loop episodes.
4. Reject policies with task failure, falls, obstacle collisions, persistent action saturation,
   excessive energy or numerical errors.
5. Exercise the simulator through the WebSocket path with injected delay/dropout.
6. Confirm malformed, replayed and stale chunks are rejected.
7. Confirm Wi-Fi loss and host death enter fallback within the configured watchdog.
8. Start the robot raised at reduced action amplitude/current.
9. Validate stand, one leg and a single gait cycle.
10. Progress to tethered floor trials before any untethered run.

VLA training loss, simulator return and a convincing video are insufficient. Use strict task
metrics and inspect the trajectory in a world-fixed view so lateral drift is visible.

## Network

Use `ws://` only on a dedicated/trusted lab LAN. The CLI does not configure certificates,
tokens or an allowlist. If control crosses an untrusted network, use a private VPN or an
authenticated TLS reverse proxy configured and tested by the operator. Never port-forward the
plain server to the internet.

## Emergency behavior

Test these deliberately while the robot is raised:

- policy process killed;
- Wi-Fi access point powered off;
- compute battery removed;
- malformed/wrong-dimensional action returned;
- action deadline exceeded;
- Orange Pi client stopped with Ctrl-C;
- Orange Pi process/control loop deliberately stalled;
- servo battery cutoff opened.

For ordinary network/process exits, the expected software outcome is no new PWM drive. For an
SBC/kernel/I2C hang, verify the independent hardware `OE`/power cutoff if fitted; otherwise the
last PCA9685 pulse can persist until the operator opens the physical cutoff. Do not choose a
stand fallback until that exact stand pose is calibrated and proven not to jam the physical
build.

## Known model gaps

- Shipped masses/inertias and MG90S response are initial estimates.
- Clone servos vary in torque, deadband, speed and pulse acceptance.
- Camera mount, exposure, distortion and IMU alignment require physical calibration.
- Printed tolerances and cable/battery placement change collision and CoM.
- A policy trained without those measurements can exploit inaccuracies despite domain
  randomization.

Treat model updates as safety-relevant changes and rerun the complete qualification suite.
