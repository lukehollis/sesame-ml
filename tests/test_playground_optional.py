from __future__ import annotations

import importlib.util

import pytest


def test_optional_package_import_does_not_require_accelerator_stack() -> None:
    import sesame_ml.playground as playground

    expected = all(
        importlib.util.find_spec(name) is not None
        for name in ("jax", "ml_collections", "mujoco_playground")
    )
    assert playground.is_available() is expected


def test_mjx_environment_contract_when_dependencies_are_installed() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    pytest.importorskip("ml_collections")
    pytest.importorskip("mujoco_playground")

    from sesame_ml.playground.environment import (
        ACTOR_OBSERVATION_SIZE,
        PRIVILEGED_OBSERVATION_SIZE,
        SesameLocomotion,
        default_config,
    )

    config = default_config()
    config.noise_config.level = 0.0
    config.impl = "jax"
    env = SesameLocomotion(config=config)
    state = jax.jit(env.reset)(jax.random.PRNGKey(7))
    assert env.action_size == 8
    assert state.obs["state"].shape == (ACTOR_OBSERVATION_SIZE,)
    assert state.obs["privileged_state"].shape == (PRIVILEGED_OBSERVATION_SIZE,)
    assert jp.allclose(state.obs["state"][12:], jp.zeros(8), atol=1e-6)

    next_state = jax.jit(env.step)(state, jp.zeros(8))
    assert jp.isfinite(next_state.reward)
    assert next_state.obs["state"].shape == (ACTOR_OBSERVATION_SIZE,)


def test_perfect_joint_state_is_privileged_when_dependencies_are_installed() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    pytest.importorskip("ml_collections")
    pytest.importorskip("mujoco_playground")

    from sesame_ml.playground.environment import SesameLocomotion, default_config

    config = default_config()
    config.noise_config.level = 0.0
    env = SesameLocomotion(config=config)
    state = env.reset(jax.random.PRNGKey(11))
    altered_qpos = state.data.qpos.at[env._joint_qpos].add(0.08)
    altered_data = state.data.replace(qpos=altered_qpos)
    original_info = dict(state.info)
    altered_info = dict(state.info)
    original = env._get_obs(state.data, original_info)
    altered = env._get_obs(altered_data, altered_info)

    # Replacing qpos alone leaves sensor data and commanded targets unchanged.
    # Therefore any direct leak of perfect q into the actor would fail this.
    assert jp.allclose(original["state"], altered["state"])
    assert not jp.allclose(original["privileged_state"], altered["privileged_state"])


def test_mjx_filters_targets_at_physical_servo_rate_when_dependencies_are_installed() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("ml_collections")
    pytest.importorskip("mujoco_playground")

    from sesame_ml.playground.environment import SesameLocomotion, default_config

    config = default_config()
    config.noise_config.level = 0.0
    env = SesameLocomotion(config=config)
    state = env.reset(jax.random.PRNGKey(23))
    info = dict(state.info)
    info["action_delay_steps"] = jp.asarray(0)
    info["servo_time_constant"] = jp.asarray(0.001)
    state = state.replace(info=info)
    previous = state.info["commanded_joints"]
    next_state = env.step(state, jp.ones(8))
    maximum_delta = config.servo_max_speed_rad_s * env.dt
    assert jp.max(jp.abs(next_state.info["commanded_joints"] - previous)) <= (
        maximum_delta + 1e-6
    )

    for name in ("obstacle_0", "obstacle_1"):
        body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        geom_ids = [
            index
            for index, geom_body_id in enumerate(env.mj_model.geom_bodyid)
            if int(geom_body_id) == body_id
        ]
        assert all(env.mj_model.geom_contype[index] == 0 for index in geom_ids)


def test_domain_randomization_batches_physical_parameters_when_installed() -> None:
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    pytest.importorskip("ml_collections")
    pytest.importorskip("mujoco_playground")

    from sesame_ml.playground.environment import SesameLocomotion, default_config
    from sesame_ml.playground.randomize import make_domain_randomizer

    env = SesameLocomotion(config=default_config())
    randomize = make_domain_randomizer(env.mj_model)
    randomized, in_axes = randomize(env.mjx_model, jax.random.split(jax.random.PRNGKey(13), 4))

    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "geom_friction",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
    ):
        assert getattr(randomized, field).shape[0] == 4
        assert getattr(in_axes, field) == 0
    assert not jp.allclose(randomized.body_mass[0], randomized.body_mass[1])
    assert not jp.allclose(randomized.actuator_gainprm[0], randomized.actuator_gainprm[1])


def test_warp_training_rejects_non_gpu_backend_when_installed() -> None:
    jax = pytest.importorskip("jax")
    pytest.importorskip("brax")
    pytest.importorskip("ml_collections")
    pytest.importorskip("mujoco_playground")
    if jax.default_backend() == "gpu":
        pytest.skip("This guard applies only when Warp has no GPU backend")

    from sesame_ml.playground.train import PPOConfig, train

    config = PPOConfig(
        impl="warp",
        num_timesteps=1,
        num_envs=1,
        num_eval_envs=1,
        require_accelerator=False,
    )
    with pytest.raises(RuntimeError, match="Warp requires a GPU backend"):
        train(config)
