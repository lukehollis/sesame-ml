# Contributing

Contributions are welcome across simulation, robot descriptions, training, evaluation,
datasets, policy integrations and hardware safety.

## Development setup

```bash
uv venv --python 3.11
uv sync --extra train --extra data --extra video --extra dev
uv run ruff check src tests scripts integrations
uv run pytest
```

Optional Playground checks require their own pinned environment:

```bash
uv sync --extra playground --extra dev
uv run pytest tests/test_playground_optional.py
```

## Change requirements

- Preserve firmware/wire joint order `[R1, R2, L1, L2, R4, R3, L3, L4]`.
- Keep normalized residual actions distinct from absolute-radian wire actions.
- Do not expose simulator-only qpos, contact, goal or velocity to a deployable actor.
- Native and MJX actuator limits/timing must remain compatible with physical servos.
- Any MJCF/URDF transform, axis, limit or mesh change must update and pass sampled FK tests.
- Any task/reward change needs a behavior-level test showing that an obvious failure policy does
  not satisfy success.
- Any hardware change must retain software timeout/shutdown behavior, accurately scope its hang
  limitations, and require explicit robot calibration.
- Do not weaken validation, limits or watchdogs to make a demonstration pass.

## Pull-request evidence

Include:

1. problem and intended behavior;
2. tests added/updated;
3. exact commands and results;
4. simulator metrics and videos for behavior changes;
5. fixed and randomized seed results;
6. hardware measurements and raised/tethered procedure for hardware-facing changes;
7. upstream commit/API version for vendor integrations.

Generated videos belong in `docs/media/` only when they document stable public behavior and are
small enough for source control. Large experiment artifacts remain ignored under `artifacts/`
or `runs/`.

## Style

The project targets Python 3.11+, Ruff's configured checks, explicit units in names, typed
dataclasses at system boundaries and machine-readable experiment artifacts. Prefer a strict
error over silently adapting wrong-dimensional or stale robot data.
