# Connecting VLAs, world models and action models

Sesame ML does not hard-code one model family. A backend receives one canonical observation
and returns a finite, calibrated action chunk. The backend can be OpenPI, GR00T, a diffusion
policy, a transformer trained on LeRobot data, a latent dynamics model with MPC, or a custom
vision-language skill planner.

## Canonical policy objects

```python
from sesame_ml.integrations import PolicyActionChunk, PolicyObservation

observation = PolicyObservation(
    state_rad=commanded_joints,      # shape (8,), absolute radians
    rgb=front_rgb,                   # uint8 H x W x 3
    instruction="walk to the green marker",
    timestamp_ns=capture_time,
    observation_seq=sequence,
)

chunk = PolicyActionChunk(
    actions_rad=targets,             # shape (horizon, 8), absolute radians
    based_on_observation_seq=sequence,
    dt_s=0.02,
    policy_name="my-world-action-model",
)
```

Construction validates shape, finiteness, time and joint ranges. The transport attaches a
freshness deadline and server-owned chunk ID. The Orange Pi performs the final calibrated
limit, interpolation, slew and watchdog checks.

## Implement a backend

```python
class MyRemotePolicy:
    def infer(self, observation):
        actions = self.model.predict(
            image=observation.rgb,
            state=observation.state_rad,
            text=observation.instruction,
        )
        return PolicyActionChunk(
            actions_rad=actions,
            based_on_observation_seq=observation.observation_seq,
            dt_s=0.02,
            policy_name="my-policy",
        )

    def reset(self):
        self.model.reset_context()

    def close(self):
        pass
```

Wrap the backend with `RemotePolicyBridge` and pass it to `PolicyWebSocketServer`, or add a
small CLI selection alongside the existing OpenPI/GR00T choices. Never truncate an arbitrary
manipulation head at the network boundary. A checkpoint-native padded head needs a named,
tested physical projection paired with its training-time padding transform: pi0/pi0.5 use an
8D-to-32D zero-pad and the exact first-eight inverse. Otherwise require a native 8D output and
reject every other shape.

## Recommended hierarchy

Large VLAs are usually slower than the robot's 50 Hz stabilization loop. Three patterns are
supported:

1. **Direct action chunks:** VLA produces 0.1-1.0 seconds of eight-joint targets. Simple, but
   balance quality depends entirely on its action data.
2. **VLA to velocity/skill:** VLA selects `forward`, `turn`, `recover`, a waypoint or a velocity
   command; PPO/MPC converts that into 50 Hz residual actions. This is the recommended starting
   point for complex tasks.
3. **World-model MPC:** learned dynamics predicts future latent/state outcomes for candidate
   action chunks; an optimizer chooses a chunk and replans after the next observation. The
   existing CEM planner is the non-learned reference implementation.

All patterns terminate at the same `ActionChunkV1` safety boundary.

## OpenPI: pi0, pi0-FAST and pi0.5

`OpenPIRemotePolicy` uses Physical Intelligence's official WebSocket client. The vendor-side
templates in `integrations/openpi/` define one front image, an eight-dimensional physical
state/action space, absolute radians, a 16-step horizon and LoRA configurations for pi0,
pi0-FAST and pi0.5. pi0/pi0.5 retain their base checkpoint's 32D internal head and use an
explicit 8D-to-32D zero-padding projection; pi0-FAST remains natively 8D.

Run OpenPI in its frozen environment and bridge its port without installing the simulator there:

```bash
PYTHONPATH=/path/to/sesame-ml/src python -m sesame_ml.bridge_cli serve \
  --policy openpi \
  --backend-host 127.0.0.1 --backend-port 8000 \
  --host 0.0.0.0 --port 8765
```

The template was checked against OpenPI commit `15a9616a` and should be reviewed when upstream
configuration APIs change. The exact frozen commands are in
[the fine-tuning guide](vla-finetuning.md).

## NVIDIA Isaac GR00T

`GrootRemotePolicy` maps RGB, eight joint values and language into GR00T's documented
batch/time modality structure and requires a `(1, horizon, 8)` action response. The vendor
templates in `integrations/groot/` register a custom `NEW_EMBODIMENT` with absolute `NON_EEF`
actions.

```bash
PYTHONPATH=/path/to/sesame-ml/src python -m sesame_ml.bridge_cli serve \
  --policy groot \
  --backend-host 127.0.0.1 --backend-port 5555 \
  --host 0.0.0.0 --port 8765
```

The template was checked against Isaac-GR00T commit `9c7e746b`. Humanoid G1/SONIC policies are
not zero-shot Sesame locomotion policies; fine-tune the custom embodiment. GR00T needs the
ephemeral WebSocket overlay shown in [the fine-tuning guide](vla-finetuning.md); do not mutate its
locked environment.

## Dataset path

Record canonical simulation episodes:

```bash
uv run sesame-ml collect \
  --env SesameNavigation-v0 \
  --policy firmware \
  --episodes 100 \
  --output artifacts/navigation-demos
```

The source episode stores lossless RGB, absolute commanded state/action, pre-action IMU and
velocity command, reward, done flags, timestamp, language, next-step
success/fall/contact/out-of-bounds/goal-distance labels, and episode domain/seed metadata in
checksummed chunks. `collect` currently records simulator policies selected by its CLI. Record
custom planner, remote-policy or physical trajectories through the same `EpisodeWriter` API,
then filter falls, timeouts, collisions, saturation and poor tracking.

Export once from OpenPI's pinned LeRobot v2.1 environment (or a dedicated v2.1 exporter) so both
vendors consume the same generation:

```bash
PYTHONPATH=/path/to/sesame-ml/src python -m sesame_ml.bridge_cli export-dataset \
  artifacts/navigation-demos/episode-* \
  --repo-id YOUR_ORG/sesame-navigation \
  --output /datasets/sesame-navigation \
  --groot-v2
```

The exact commit checks, dedicated-exporter alternative and official GR00T v3-to-v2 fallback are
in [the fine-tuning guide](vla-finetuning.md). Split by whole trajectory and randomized scene
seed, never by individual frames. The behavior-cloning LeRobot export contains front RGB,
state/action, reward/done and language; richer IMU/world-model labels remain in the canonical
source episode.

## World-model data

A learned dynamics or world-action model generally needs more context than behavior cloning.
Schema v2 records the context needed to assemble windows containing:

- several RGB/state/action frames before the prediction time;
- exact action timestamps and control period;
- pre-action IMU and velocity/task context;
- later RGB/state rows, reward and next-step success/fall/collision/out-of-bounds labels;
- episode domain parameters and scene seed.

Future RGB/state are formed by shifting rows within a trajectory; there is no post-terminal
frame. Physical collection is an API integration rather than a shipped camera-recording CLI, so
record robot calibration and hardware revision in `EpisodeWriter` metadata when adding it.

Keep the model backend out of the Orange Pi process. Serve inference on the CUDA machine and
return only finite, deadline-bounded action chunks.

## Evaluation sequence

1. Validate action shapes/ranges and offline held-out errors.
2. Run closed-loop fixed-scene simulation.
3. Run domain-randomized and camera/latency-shifted simulation.
4. Exercise the real WebSocket path with delay/dropout injection.
5. Test raised hardware at reduced amplitude.
6. Run tethered floor trials with a cutoff.
7. Only then perform untethered task evaluation.

Report task success, falls, collisions, tracking/goal error, energy/action rate, inference and
round-trip latency, rejected/stale chunks and watchdog entries. Training loss is not a robot
task metric.
