from __future__ import annotations

import json

import pytest

pytest.importorskip("stable_baselines3")

from sesame_ml.training import PPOConfig, train_ppo  # noqa: E402


def _small_config(tmp_path, name: str, **updates) -> PPOConfig:  # type: ignore[no-untyped-def]
    values = {
        "task": "stand",
        "total_timesteps": 16,
        "number_of_environments": 1,
        "seed": 19,
        "output_directory": tmp_path,
        "run_name": name,
        "domain_randomization": False,
        "evaluation_domain_randomization": False,
        "maximum_episode_steps": 8,
        "normalize_observations": True,
        "normalize_rewards": True,
        "rollout_steps": 8,
        "batch_size": 8,
        "epochs": 1,
        "network_width": 32,
        "network_depth": 1,
        "checkpoint_frequency": 8,
        "evaluation_frequency": 8,
        "evaluation_episodes": 1,
        "verbose": 0,
    }
    values.update(updates)
    return PPOConfig(**values)


def test_ppo_training_checkpoints_and_resume(tmp_path) -> None:
    first = train_ppo(_small_config(tmp_path, "initial"))
    assert first.samples_seen == 16
    assert first.final_checkpoint.exists()
    assert first.best_checkpoint is not None and first.best_checkpoint.exists()
    assert (first.best_checkpoint.parent / "vecnormalize.pkl").exists()
    assert first.vecnormalize_path is not None and first.vecnormalize_path.exists()
    assert list((first.run_directory / "checkpoints").glob("*.zip"))
    status = json.loads((first.run_directory / "status.json").read_text())
    assert status["status"] == "complete"

    resumed = train_ppo(
        _small_config(
            tmp_path,
            "resumed",
            total_timesteps=8,
            # Exercise automatic pairing with CheckpointCallback's step-specific
            # sesame_ppo_vecnormalize_16_steps.pkl artifact.
            resume_from=(
                first.run_directory / "checkpoints" / "sesame_ppo_16_steps.zip"
            ),
        )
    )
    assert resumed.resumed
    assert resumed.samples_seen == first.samples_seen + 8
    assert resumed.final_checkpoint.exists()
