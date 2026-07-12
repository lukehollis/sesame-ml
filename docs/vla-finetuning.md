# OpenPI and GR00T fine-tuning

This document covers vendor-specific supervised imitation learning. For the generic backend
interface, skill hierarchy and world-model path, start with
[VLA and world-model integration](vla-world-models.md).

A manipulation or humanoid checkpoint is not a zero-shot Sesame gait. Use a custom
eight-dimensional embodiment and qualified demonstrations.

## 1. Record and qualify trajectories

```bash
uv run sesame-ml collect \
  --env SesameNavigation-v0 \
  --policy firmware \
  --episodes 100 \
  --output artifacts/navigation-demos
```

Add successful PPO, MPC and later real trajectories. Reject falls, timeouts, out-of-bounds
episodes, obstacle contact, persistent saturation and poor tracking. Split by trajectory and
scene/domain seed, not by frame.

The canonical checksummed episode is the source of truth. Vendor LeRobot datasets are derived
training artifacts.

## 2. Export once with pinned LeRobot v2.1

OpenPI and GR00T use incompatible environments. Do not install the full Sesame simulator into
either, and do not mutate a vendor lock with `uv pip install`. OpenPI commit `15a9616a` already
locks the bridge dependencies and LeRobot v2.1 commit `0cf86487`, so it is the preferred single
exporter for both model families:

```bash
cd OPENPI_CHECKOUT
git rev-parse HEAD | grep -Fx 15a9616a00943ada6c20a0f158e3adb39df2ccac
uv sync --frozen

PYTHONPATH=/absolute/path/to/sesame-ml/src \
uv run --frozen --no-sync python -m sesame_ml.bridge_cli export-dataset \
  /absolute/path/to/navigation-demos/episode-* \
  --repo-id YOUR_ORG/sesame-navigation-v2 \
  --output /datasets/sesame-navigation-v2 \
  --groot-v2
```

`--groot-v2` requires MP4-backed video, writes `meta/modality.json`, and rejects any result that
is not a complete LeRobot v2 dataset. Use this same dataset for OpenPI and GR00T.

For a GR00T-only setup, create a dedicated exporter environment rather than changing GR00T's
environment:

```bash
uv venv /tmp/sesame-export --python 3.11
VIRTUAL_ENV=/tmp/sesame-export uv pip install \
  -r /absolute/path/to/sesame-ml/integrations/bridge-requirements.txt \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

PYTHONPATH=/absolute/path/to/sesame-ml/src \
/tmp/sesame-export/bin/python -m sesame_ml.bridge_cli export-dataset \
  /absolute/path/to/navigation-demos/episode-* \
  --repo-id YOUR_ORG/sesame-navigation-v2 \
  --output /datasets/sesame-navigation-v2 \
  --groot-v2
```

The official alternative is to export v3 in GR00T's separately locked
`scripts/lerobot_conversion` environment, run
`python convert_v3_to_v2.py --repo-id YOUR_ORG/sesame-navigation --root /datasets`, then call
`write_groot_metadata()` and `validate_groot_v2_dataset()`. The converter modifies the v2 dataset
in place under `/datasets/YOUR_ORG/sesame-navigation` and retains a sibling `_v3.0` backup.

## 3. OpenPI: pi0, pi0-FAST and pi0.5

The source templates live in `integrations/openpi/` and are also installed under
`sesame_ml/vendor_templates/openpi/`. They were checked against OpenPI commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac`; compare them with the exact upstream checkout you
install.

Copy:

```text
integrations/openpi/sesame_policy.py
    -> OPENPI/src/openpi/policies/sesame_policy.py

integrations/openpi/sesame_train_configs.py
    -> OPENPI/src/openpi/training/sesame_train_configs.py
```

After OpenPI builds `_CONFIGS = [...]`, but before its duplicate-name check and `_CONFIGS_DICT`,
register the tuple with `extend` (not `append`):

```python
from openpi.training import sesame_train_configs as _sesame_train_configs  # noqa: E402

_CONFIGS.extend(
    _sesame_train_configs.make_sesame_configs("YOUR_ORG/sesame-navigation-v2")
)
```

The factory defines pi0, pi0-FAST and pi0.5 LoRA runs with:

- one real front image and explicit padded/masked vendor views;
- eight physical absolute-radian state/action values;
- explicit zero-padding to the 32D `pi0_base`/`pi05_base` checkpoint projection and an explicit
  first-eight output projection after denormalization; pi0-FAST remains natively 8D;
- a 16-step horizon;
- no manipulator delta-action transform;
- strict action-head dimensionality.

Start with pi0.5:

```bash
uv run --frozen --no-sync scripts/compute_norm_stats.py --config-name pi05_sesame_lora

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run --frozen --no-sync scripts/train.py pi05_sesame_lora \
  --exp-name=sesame-navigation --overwrite

uv run --frozen --no-sync scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_sesame_lora \
  --policy.dir=checkpoints/pi05_sesame_lora/sesame-navigation/30000
```

Bridge OpenPI's local port 8000 from that same pinned environment:

```bash
PYTHONPATH=/absolute/path/to/sesame-ml/src \
uv run --frozen --no-sync python -m sesame_ml.bridge_cli serve \
  --policy openpi \
  --backend-host 127.0.0.1 --backend-port 8000 \
  --host 0.0.0.0 --port 8765
```

Run `sim-client` from the normal Sesame simulator environment before `orange-client`. The vendor
transform explicitly maps the verified 32D pi0/pi0.5 head back to the first eight padded physical
dimensions; the network adapter still rejects anything other than an unbatched `(horizon, 8)`
absolute-radian result.

The template source and full transform shapes were checked against the pinned OpenPI API. This
release has not downloaded all three multi-gigabyte base checkpoints in CI. Before committing a
training run, load the actual selected base checkpoint, compute normalization statistics, and run
one forward/loss batch; OpenPI's loader must report no missing or mismatched parameter shapes.

OpenPI's JAX path is required for pi0-FAST and LoRA in the tested stack. Keep it separate from
the MuJoCo and GR00T environments.

## 4. NVIDIA Isaac GR00T N1.7

Templates live in `integrations/groot/` and the installed `sesame_ml/vendor_templates/groot/`.
They were checked against Isaac-GR00T commit
`9c7e746b2cd37a810070a98ef41d290a07e806c2` and preserve NVIDIA's Apache-2.0 SPDX notice.

The configuration registers `NEW_EMBODIMENT`, one front video, eight joint-state values,
16 absolute `NON_EEF` joint targets and the human task description.

Verify the checkout and environment, then run the official statistics preflight before
fine-tuning:

```bash
git rev-parse HEAD | grep -Fx 9c7e746b2cd37a810070a98ef41d290a07e806c2
uv sync --frozen
uv run --frozen --no-sync python gr00t/data/stats.py \
  --dataset-path /datasets/sesame-navigation-v2 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /absolute/path/to/sesame_config.py
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen --no-sync python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /datasets/sesame-navigation-v2 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /path/to/sesame_config.py \
  --num-gpus 1 \
  --output-dir /checkpoints/sesame-groot \
  --max-steps 10000 \
  --global-batch-size 32 \
  --dataloader-num-workers 4
```

Run vendor open-loop evaluation first:

```bash
uv run --frozen --no-sync python gr00t/eval/open_loop_eval.py \
  --dataset-path /datasets/sesame-navigation-v2 \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /checkpoints/sesame-groot/checkpoint-10000 \
  --traj-ids 0 1 2 \
  --modality-keys joints \
  --execution-horizon 16
```

Then serve and bridge:

```bash
uv run --frozen --no-sync python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /checkpoints/sesame-groot/checkpoint-10000 \
  --device cuda:0 --host 127.0.0.1 --port 5555

PYTHONPATH=/absolute/path/to/sesame-ml/src \
uv run --frozen --no-sync --with 'websockets>=14,<17' \
  python -m sesame_ml.bridge_cli serve \
  --policy groot \
  --backend-host 127.0.0.1 --backend-port 5555 \
  --host 0.0.0.0 --port 8765
```

GR00T's humanoid whole-body paths are not substitutes for the custom Sesame embodiment.

## 5. Evaluation gates

For each checkpoint:

1. Held-out action plots, dimensional/range validation and offline error.
2. Fixed-scene closed-loop simulation.
3. Domain-randomized simulation.
4. Network-path latency/dropout/watchdog evaluation.
5. Raised hardware with reduced amplitude/current.
6. Tethered floor trials with a cutoff.
7. Untethered trials only after all prior gates pass.

Report task success, fall/out-of-bounds rate, tracking and final goal error, obstacle contact,
action smoothness/saturation, energy, inference/round-trip latency, rejected/stale chunks and
watchdog entries. VLA loss alone is not a robot policy metric.
