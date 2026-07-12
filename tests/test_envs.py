from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import sesame_ml.envs  # noqa: F401
from sesame_ml.envs import SesameEnv, SesameEnvConfig, Task


@pytest.mark.parametrize(
    "environment_id",
    ["SesameStand-v0", "SesameRecovery-v0", "SesameLocomotion-v0", "SesameNavigation-v0"],
)
def test_registered_environment_passes_basic_contract(environment_id: str) -> None:
    env = gym.make(environment_id, domain_randomization=False)
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert info["task"] in environment_id.lower()
    for _ in range(10):
        observation, reward, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()


def test_feedback_free_observation_hides_true_joint_velocity() -> None:
    env = SesameEnv(
        task=Task.LOCOMOTION,
        config=SesameEnvConfig(task=Task.LOCOMOTION, domain_randomization=False),
    )
    observation, _ = env.reset(seed=2)
    assert np.linalg.norm(observation[3:6]) > 0.5  # normalized physical accelerometer
    assert np.array_equal(observation[17:25], np.zeros(8, dtype=np.float32))
    assert np.array_equal(observation[36:43], np.zeros(7, dtype=np.float32))
    env.close()


def test_privileged_observation_explicitly_exposes_contacts_and_goal() -> None:
    env = SesameEnv(
        task=Task.NAVIGATION,
        config=SesameEnvConfig(
            task=Task.NAVIGATION,
            domain_randomization=False,
            privileged_joint_state=True,
        ),
    )
    observation, _ = env.reset(seed=2)
    assert np.any(observation[36:40] > 0)
    assert np.linalg.norm(observation[40:43]) > 0.5
    env.close()


def test_pixel_navigation_observation_matches_camera_space() -> None:
    env = SesameEnv(
        task=Task.NAVIGATION,
        config=SesameEnvConfig(
            task=Task.NAVIGATION,
            observation_mode="pixels",
            camera_width=96,
            camera_height=72,
            domain_randomization=False,
        ),
    )
    observation, info = env.reset(seed=4)
    assert observation["rgb"].shape == (72, 96, 3)
    assert observation["rgb"].dtype == np.uint8
    assert "green marker" in info["instruction"]
    env.close()


def test_pixel_observation_holds_frames_at_deployment_camera_rate() -> None:
    env = SesameEnv(
        task=Task.LOCOMOTION,
        config=SesameEnvConfig(
            task=Task.LOCOMOTION,
            observation_mode="pixels",
            camera_width=64,
            camera_height=48,
            camera_hz=10,
            domain_randomization=False,
        ),
    )
    initial, _ = env.reset(seed=4)
    first, *_ = env.step(np.ones(8, dtype=np.float32) * 0.2)
    second, *_ = env.step(np.ones(8, dtype=np.float32) * -0.2)
    assert np.array_equal(initial["rgb"], first["rgb"])
    assert np.array_equal(initial["rgb"], second["rgb"])
    for _ in range(3):
        second, *_ = env.step(np.ones(8, dtype=np.float32) * 0.2)
    assert not np.array_equal(initial["rgb"], second["rgb"])
    env.close()


def test_pixel_and_external_video_use_persistent_separate_renderers() -> None:
    env = SesameEnv(
        task=Task.NAVIGATION,
        config=SesameEnvConfig(
            task=Task.NAVIGATION,
            observation_mode="pixels",
            camera_width=96,
            camera_height=72,
            render_width=160,
            render_height=120,
            domain_randomization=False,
            maximum_episode_steps=2,
        ),
        render_mode="rgb_array",
    )
    env.reset(seed=4)
    camera_renderer = env._camera_renderer
    assert camera_renderer is not None
    assert env.render().shape == (120, 160, 3)
    external_renderer = env._external_renderer
    assert external_renderer is not None
    assert external_renderer is not camera_renderer

    env.step(np.zeros(8, dtype=np.float32))
    env.render()
    assert env._camera_renderer is camera_renderer
    assert env._external_renderer is external_renderer
    env.close()


def test_absolute_servo_action_path() -> None:
    env = SesameEnv(task=Task.STAND, domain_randomization=False)
    env.reset(seed=1)
    target = STAND = env.servo_targets
    _, reward, terminated, _, info = env.step_servo_targets(target)
    assert reward > 0
    assert not terminated
    assert np.allclose(info["servo_target_rad"], STAND, atol=1e-6)
    env.close()


def test_navigation_props_are_disabled_outside_navigation() -> None:
    for task in (Task.STAND, Task.RECOVERY, Task.LOCOMOTION):
        env = SesameEnv(task=task, domain_randomization=False)
        env.reset(seed=7)
        geom_ids = env._navigation_geom_ids
        assert np.all(env.model.geom_rgba[geom_ids, 3] == 0)
        assert np.all(env.model.geom_contype[geom_ids] == 0)
        assert np.all(env.model.geom_conaffinity[geom_ids] == 0)
        env.close()

    env = SesameEnv(task=Task.NAVIGATION, domain_randomization=False)
    env.reset(seed=7)
    geom_ids = env._navigation_geom_ids
    assert np.all(env.model.geom_rgba[geom_ids, 3] > 0)
    obstacle_ids = geom_ids[1:]
    assert np.all(env.model.geom_contype[obstacle_ids] != 0)
    assert np.all(env.model.geom_conaffinity[obstacle_ids] != 0)
    env.close()


def test_randomized_position_servo_keeps_gain_and_bias_paired() -> None:
    env = SesameEnv(task=Task.STAND, domain_randomization=True)
    observation, _ = env.reset(seed=17)
    assert np.allclose(
        env.model.actuator_gainprm[:, 0], -env.model.actuator_biasprm[:, 1]
    )
    policy_action = np.zeros(8, dtype=np.float32)
    for _ in range(50):
        observation, _, terminated, truncated, _ = env.step(policy_action)
        assert not terminated and not truncated
    assert np.max(np.abs(env.true_joint_positions - env.servo_targets)) < np.deg2rad(1)
    env.close()


def test_reset_contact_penetration_is_submillimeter_and_recovery_is_challenging() -> None:
    upright = SesameEnv(task=Task.STAND, domain_randomization=False)
    upright.reset(seed=3)
    floor_penetrations = [
        -float(contact.dist)
        for contact in upright.data.contact[: upright.data.ncon]
        if contact.dist < 0
    ]
    assert max(floor_penetrations, default=0.0) < 0.001
    upright.close()

    recovery = SesameEnv(task=Task.RECOVERY, domain_randomization=False)
    _, info = recovery.reset(seed=3)
    assert info["upright"] < 0.9
    assert not info["success"]
    recovery.close()


def test_navigation_out_of_bounds_is_failure_but_not_fall() -> None:
    env = SesameEnv(task=Task.NAVIGATION, domain_randomization=False)
    env.reset(seed=3)
    env.data.qpos[env._root_qpos_adr] = 3.1
    observation, reward, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert terminated and not truncated
    assert info["out_of_bounds"]
    assert not info["episode"]["fall"]
    env.close()
