# Getting started

## Requirements

- Python 3.11 or 3.12.
- macOS or Linux for native MuJoCo simulation.
- Ubuntu plus a supported NVIDIA/CUDA stack for large MJX/Warp or vendor VLA training.
- `uv` is recommended; standard `pip` environments also work.

The package contains its own MJCF, URDF and STL assets. It can be installed and run outside a
Sesame Robot source checkout.

## Choose an environment

Base simulator and CLI:

```bash
uv venv --python 3.11
uv sync
uv run sesame-ml validate-model
```

Local PPO, datasets, videos and development checks:

```bash
uv sync --extra train --extra data --extra video --extra dev
uv run pytest
```

Orange Pi hardware runtime:

```bash
uv sync --extra hardware
```

CUDA-scale Playground training belongs in a separate environment:

```bash
uv venv --python 3.12
uv sync --extra playground-cuda --extra dev
```

OpenPI and GR00T should each have their own vendor environment or container. Their JAX,
PyTorch, CUDA and LeRobot versions are not expected to coexist with one another.

## Validate the model

```bash
uv run sesame-ml validate-model --seconds 5
```

This compiles the packaged model, applies the firmware stand pose and executes real dynamics.
It reports actuator/body/geometry counts, dynamic mass, body height, joint error and contacts.

Open an interactive viewer:

```bash
uv run sesame-ml view --env SesameStand-v0 --policy stand
uv run sesame-ml view --env SesameLocomotion-v0 --policy firmware
```

On macOS the viewer automatically re-launches through MuJoCo's `mjpython` executable.

## Record an evaluation

```bash
uv run sesame-ml evaluate \
  --env SesameStand-v0 \
  --env SesameRecovery-v0 \
  --env SesameLocomotion-v0 \
  --policy cpg \
  --episodes 10 \
  --videos 2 \
  --output artifacts/baseline-eval
```

Use `--video-view external`, `front`, or `split`. `front` records the exact held RGB frame a
pixel policy receives; `split` places that frame inside the task-aware external view.

The output contains:

- `report.json` with configuration, summaries and every episode;
- `episodes.csv` for analysis;
- H.264 MP4 files for the selected episodes.

Locomotion success requires command tracking, not merely remaining upright. Recovery starts
outside the success threshold and requires a sustained stable pose.

## Train and evaluate PPO

```bash
uv run sesame-ml train-ppo \
  --env SesameLocomotion-v0 \
  --steps 2000000 \
  --num-envs 8 \
  --output runs/locomotion
```

Training writes periodic, best, interrupted and final checkpoints, paired VecNormalize state,
TensorBoard logs, evaluation arrays and a machine-readable status file. A short run validates
the pipeline only; it does not establish convergence.

```bash
uv run sesame-ml evaluate \
  --env SesameLocomotion-v0 \
  --policy ppo \
  --checkpoint runs/locomotion/best/best_model.zip \
  --vecnormalize runs/locomotion/best/vecnormalize.pkl \
  --episodes 50 \
  --videos 3 \
  --output artifacts/locomotion-final
```

## Exercise remote inference safely

Run a local baseline server:

```bash
uv run sesame-ml serve --policy cpg --host 127.0.0.1 --port 8765
```

In a second terminal, make MuJoCo behave like the Wi-Fi robot:

```bash
uv run sesame-ml sim-client \
  --uri ws://127.0.0.1:8765 \
  --env SesameNavigation-v0
```

Only after testing disconnect, stale-chunk and watchdog behavior should the URI be changed to
the policy machine's private-LAN address and used by `orange-client`.

The CLI server is plain `ws://` and intended for a trusted lab LAN. It is not an internet
endpoint. Put a real authenticated TLS reverse proxy or VPN in front of it if traffic leaves
that environment.

## Next documents

- [Simulation and evaluation](simulation.md)
- [Architecture and deployment contract](architecture.md)
- [VLA and world-model integration](vla-world-models.md)
- [OpenPI and GR00T fine-tuning](vla-finetuning.md)
- [Orange Pi reference hardware](hardware-reference.md)
- [URDF validation](urdf.md)
- [Safety and sim-to-real](safety.md)
