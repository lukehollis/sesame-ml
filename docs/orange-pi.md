# Orange Pi deployment

This procedure deploys the robot-side client after simulation and network-path evaluation.
The reference build has one Orange Pi Zero 3W computer, a USB UVC camera, BNO085 IMU and a
PCA9685 PWM peripheral. Policy/VLA inference remains on a workstation.

Read the illustrated [two-battery hardware reference](hardware-reference.md) and
[safety checklist](safety.md) first.

## Install

Use an actively supported Linux image for the exact board:

```bash
sudo apt update
sudo apt install -y python3-dev python3-venv libgl1 i2c-tools v4l-utils

cd sesame-ml
uv venv --python 3.11
uv sync --extra hardware
```

## Identify camera and I2C

Do this with servo `V+` disconnected:

```bash
v4l2-ctl --list-devices
v4l2-ctl --device /dev/video0 --list-formats-ext
v4l2-ctl --device /dev/video0 --all

i2cdetect -l
i2cdetect -y ACTUAL_BUS
```

Default addresses in the examples are PCA9685 `0x40` and BNO085 `0x4a`; verify the installed
breakouts. The CLI's I2C bus default is only a placeholder. Enable the desired TWI/I2C bus in
the exact Orange Pi device tree and use the board revision's official pinout.

The public reference uses USB UVC/V4L2. An Orange Pi 5 CSI camera is not assumed compatible
with a Zero 3W connector, driver or device tree.

## Create a required robot calibration

`orange-client` refuses to start without `--calibration`. Copy the example as a template, then
measure every value on the raised/current-limited robot:

```bash
cp src/sesame_ml/assets/calibration/default.yaml ~/sesame-001.yaml
```

Record:

- exact joint order `[R1, R2, L1, L2, R4, R3, L3, L4]`;
- each mirrored direction;
- subtrim at a printed alignment pose;
- minimum/maximum angle with mechanical and self-collision margin;
- verified PWM minimum/maximum for the installed servos;
- maximum command step and speed;
- action watchdog and final disable timeout.

Only after completing those checks, set `calibrated: true`; the untouched template is rejected.

The loader consumes every `safety` field. At each 50 Hz write, the tighter of
`maximum_step_degrees` and `maximum_speed_degrees_s / 50` is applied. `action_timeout_ms`
controls network fallback. When `--fallback stand` is selected, the already-calibrated stand is
held only until `disable_timeout_ms`, after which PWM is disabled.

Generic 732-2929 microsecond endpoints are not proof that a servo accepts that range. Never
guess directions or endpoints from a photo.

## Test the server with MuJoCo

On the policy workstation:

```bash
uv run sesame-ml serve --policy stand --host 0.0.0.0 --port 8765
```

From another process/machine, before hardware:

```bash
uv run sesame-ml sim-client \
  --uri ws://POLICY_HOST:8765 \
  --env SesameStand-v0
```

Kill the server and disconnect Wi-Fi while inspecting transport/watchdog metrics. The CLI uses
plain `ws://` and is intended only for a dedicated trusted LAN. Use a tested VPN or authenticated
TLS reverse proxy outside that boundary; the CLI does not configure internet-grade auth/TLS.

## Run the robot client

Keep the robot raised and servo battery disarmed. Start compute, camera, IMU and the outbound
client:

```bash
uv run sesame-ml orange-client \
  --uri ws://192.168.1.20:8765 \
  --robot-id sesame-001 \
  --camera /dev/video0 \
  --i2c-bus 3 \
  --pca9685-address 0x40 \
  --imu-address 0x4a \
  --calibration ~/sesame-001.yaml \
  --fallback disable
```

Use `disable` until the exact physical stand is validated. `stand` is not inherently safer if
the calibration/directions are wrong or a leg is trapped.

## Power-on sequence

1. Servo branch off; robot raised; cutoff reachable.
2. Start compute branch and client; verify camera/IMU observations on the host.
3. Confirm the PCA9685 full-off transaction on timeout and client exit. If the build requires
   hang-tolerant shutdown, separately verify an independent watchdog driving PCA9685 `OE` or the
   rated servo-power cutoff.
4. Arm a current-limited servo supply with the policy in Stand.
5. Enable and verify one servo/leg at a time.
6. Verify complete Stand, then one low-amplitude gait cycle.
7. Repeat server-death, Wi-Fi-loss and compute-battery-loss tests.
8. Progress to tethered floor trials only after current/voltage/temperature remain within the
   measured power budget.

The runtime requests PCA9685 full-off on disconnect, timeout, orderly shutdown or a `None`
target, and it never drains an unbounded queue of stale PWM commands. That software action
requires the process, kernel and I2C bus to remain operational; it does not cover an SBC,
kernel, control-loop or I2C hang. Use the independent hardware `OE`/servo-power watchdog from
the reference design when automatic hang response is required, and always retain a reachable
physical cutoff.
