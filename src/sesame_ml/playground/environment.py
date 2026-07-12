"""GPU-vectorizable Sesame locomotion environment for MuJoCo Playground."""

from __future__ import annotations

from typing import Any

try:
    import jax
    import jax.numpy as jp
    import mujoco
    from ml_collections import config_dict
    from mujoco import mjx
    from mujoco_playground._src import mjx_env
except ImportError as exc:  # pragma: no cover - exercised by the lazy package boundary.
    raise ImportError(
        "sesame_ml.playground.environment requires JAX, ml-collections, and "
        "the MuJoCo Playground package; install sesame-ml[playground] or "
        "sesame-ml[playground-cuda]."
    ) from exc

from sesame_ml.constants import (
    JOINT_LIMITS_RAD,
    JOINT_NAMES,
    SERVO_MAX_SPEED_RAD_S,
    STAND_ANGLES_RAD,
)
from sesame_ml.model import (
    joint_dof_addresses,
    joint_qpos_addresses,
    load_model,
    mjcf_path,
)
from sesame_ml.model import make_data as make_cpu_data

ACTOR_OBSERVATION_SIZE = 20
PRIVILEGED_OBSERVATION_SIZE = 61


def default_config() -> config_dict.ConfigDict:
    """Return the environment defaults used by the production PPO entrypoint."""

    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1_000,
        action_scale=0.58,
        servo_max_speed_rad_s=SERVO_MAX_SPEED_RAD_S,
        servo_time_constant_seconds=[0.025, 0.080],
        maximum_action_delay_steps=3,
        command_resample_seconds=[2.5, 6.0],
        command_min=[-0.12, -0.06, -1.2],
        command_max=[0.24, 0.06, 1.2],
        stopped_command_probability=0.15,
        reset_joint_noise=0.035,
        reset_velocity_noise=0.05,
        noise_config=config_dict.create(
            level=1.0,
            gravity=0.01,
            gyro=0.015,
            accelerometer=0.03,
        ),
        reward_config=config_dict.create(
            tracking_sigma=0.10,
            yaw_tracking_sigma=0.45,
            pose_sigma=0.30,
            scales=config_dict.create(
                tracking_linear_velocity=1.50,
                tracking_yaw_velocity=0.55,
                pose=0.25,
                upright=0.50,
                energy=-0.003,
                action_rate=-0.025,
                termination=-2.0,
            ),
        ),
        termination_height=0.023,
        termination_upright=0.15,
        impl="jax",
        # Warp allocates these across the vectorized worlds.  The train entrypoint
        # increases naconmax when num_envs exceeds this production default.
        naconmax=8 * 8_192,
        naccdmax=8 * 8_192,
        njmax=64,
    )


def _quat_multiply(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Multiply scalar-first MuJoCo quaternions without a private JAX API."""

    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return jp.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _require_model_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Sesame model is missing required {object_type.name} {name!r}")
    return object_id


class SesameLocomotion(mjx_env.MjxEnv):
    """Command-conditioned locomotion with deployable actor observations.

    Actions are eight residual servo targets in canonical firmware order:
    ``R1, R2, L1, L2, R4, R3, L3, L4``.  The actor receives only quantities
    available on a feedback-free physical Sesame: IMU-derived signals, the
    velocity command, and the last commanded joint targets.  Perfect MuJoCo
    positions, velocities, forces, and contacts are confined to the asymmetric
    critic observation under ``privileged_state``.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config or default_config(), config_overrides)
        if self._config.impl not in {"jax", "warp"}:
            raise ValueError("impl must be 'jax' or 'warp'")

        self._mj_model = load_model()
        # The shared MJCF carries navigation props, but this environment is locomotion-only.
        # Disable them before conversion so thousands of MJX worlds do not contain a hidden
        # obstacle course.
        obstacle_body_ids = {
            _require_model_id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_0"),
            _require_model_id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_1"),
        }
        target_geom_id = _require_model_id(
            self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, "target_visual"
        )
        navigation_geom_ids = [target_geom_id]
        navigation_geom_ids.extend(
            index
            for index, body_id in enumerate(self._mj_model.geom_bodyid)
            if int(body_id) in obstacle_body_ids
        )
        self._mj_model.geom_rgba[navigation_geom_ids, 3] = 0
        self._mj_model.geom_contype[navigation_geom_ids] = 0
        self._mj_model.geom_conaffinity[navigation_geom_ids] = 0
        self._mj_model.opt.timestep = self._config.sim_dt
        try:
            self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        except (AttributeError, RuntimeError) as exc:
            if self._config.impl != "warp":
                raise
            raise RuntimeError(
                "MuJoCo Warp could not initialize. Warp training requires a "
                "supported NVIDIA GPU, CUDA-enabled JAX, and mutually compatible "
                "MuJoCo MJX and warp-lang versions."
            ) from exc
        self._xml_path = str(mjcf_path())

        initial = make_cpu_data(self._mj_model, settle=True)
        self._initial_qpos = jp.asarray(initial.qpos)
        self._initial_ctrl = jp.asarray(STAND_ANGLES_RAD)
        self._stand = jp.asarray(STAND_ANGLES_RAD)
        self._joint_low = jp.asarray(JOINT_LIMITS_RAD[:, 0])
        self._joint_high = jp.asarray(JOINT_LIMITS_RAD[:, 1])
        self._joint_qpos = jp.asarray(joint_qpos_addresses(self._mj_model))
        self._joint_dof = jp.asarray(joint_dof_addresses(self._mj_model))

        root_id = _require_model_id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self._root_qpos = int(self._mj_model.jnt_qposadr[root_id])
        self._root_dof = int(self._mj_model.jnt_dofadr[root_id])
        self._imu_site = _require_model_id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self._sensor_slices = {}
        for name in (
            "imu_gyro",
            "imu_accel",
            "imu_velocity",
            "r3_touch",
            "r4_touch",
            "l3_touch",
            "l4_touch",
        ):
            sensor_id = _require_model_id(self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self._sensor_slices[name] = (
                int(self._mj_model.sensor_adr[sensor_id]),
                int(self._mj_model.sensor_dim[sensor_id]),
            )
        self._command_min = jp.asarray(self._config.command_min)
        self._command_max = jp.asarray(self._config.command_max)
        self._command_scale = jp.maximum(jp.abs(self._command_min), jp.abs(self._command_max))

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return len(JOINT_NAMES)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def _make_data(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        kwargs: dict[str, Any] = {}
        if self._config.impl == "warp":
            kwargs = {
                "naconmax": self._config.naconmax,
                "naccdmax": self._config.naccdmax,
                "njmax": self._config.njmax,
            }
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=self._initial_ctrl,
            impl=self._config.impl,
            **kwargs,
        )
        return mjx.forward(self.mjx_model, data)

    def _sensor(self, data: mjx.Data, name: str) -> jax.Array:
        address, dimension = self._sensor_slices[name]
        return data.sensordata[address : address + dimension]

    def _projected_gravity(self, data: mjx.Data) -> jax.Array:
        rotation = data.site_xmat[self._imu_site]
        return rotation.T @ jp.array([0.0, 0.0, -1.0])

    def _up_vector(self, data: mjx.Data) -> jax.Array:
        rotation = data.site_xmat[self._imu_site]
        return rotation.T @ jp.array([0.0, 0.0, 1.0])

    def _contacts(self, data: mjx.Data) -> jax.Array:
        return jp.array(
            [self._sensor(data, f"{foot}_touch")[0] > 1e-5 for foot in ("r3", "r4", "l3", "l4")]
        )

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        command_rng, stop_rng = jax.random.split(rng)
        command = jax.random.uniform(
            command_rng, shape=(3,), minval=self._command_min, maxval=self._command_max
        )
        stopped = jax.random.bernoulli(stop_rng, p=self._config.stopped_command_probability)
        return jp.where(stopped, jp.zeros(3), command)

    def _sample_command_steps(self, rng: jax.Array) -> jax.Array:
        seconds = jax.random.uniform(
            rng,
            minval=self._config.command_resample_seconds[0],
            maxval=self._config.command_resample_seconds[1],
        )
        return jp.maximum(1, jp.round(seconds / self.dt).astype(jp.int32))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (
            rng,
            xy_rng,
            yaw_rng,
            joint_rng,
            velocity_rng,
            command_rng,
            time_rng,
            delay_rng,
            servo_rng,
        ) = jax.random.split(rng, 9)
        qpos = self._initial_qpos
        qpos = qpos.at[self._root_qpos : self._root_qpos + 2].add(
            jax.random.uniform(xy_rng, (2,), minval=-0.25, maxval=0.25)
        )
        yaw = jax.random.uniform(yaw_rng, minval=-jp.pi, maxval=jp.pi)
        yaw_quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
        root_quat = qpos[self._root_qpos + 3 : self._root_qpos + 7]
        qpos = qpos.at[self._root_qpos + 3 : self._root_qpos + 7].set(
            _quat_multiply(root_quat, yaw_quat)
        )
        joint_noise = jax.random.uniform(
            joint_rng,
            (len(JOINT_NAMES),),
            minval=-self._config.reset_joint_noise,
            maxval=self._config.reset_joint_noise,
        )
        qpos = qpos.at[self._joint_qpos].set(
            jp.clip(self._stand + joint_noise, self._joint_low, self._joint_high)
        )
        qvel = jax.random.uniform(
            velocity_rng,
            (self.mjx_model.nv,),
            minval=-self._config.reset_velocity_noise,
            maxval=self._config.reset_velocity_noise,
        )
        data = self._make_data(qpos, qvel)
        info = {
            "rng": rng,
            "command": self._sample_command(command_rng),
            "commanded_joints": self._stand,
            "last_action": jp.zeros(len(JOINT_NAMES)),
            "steps_until_command": self._sample_command_steps(time_rng),
            "target_buffer": jp.tile(
                self._stand[None, :], (int(self._config.maximum_action_delay_steps) + 1, 1)
            ),
            "action_delay_steps": jax.random.randint(
                delay_rng,
                (),
                0,
                int(self._config.maximum_action_delay_steps) + 1,
            ),
            "servo_time_constant": jax.random.uniform(
                servo_rng,
                (),
                minval=self._config.servo_time_constant_seconds[0],
                maxval=self._config.servo_time_constant_seconds[1],
            ),
        }
        metrics = {f"reward/{name}": jp.zeros(()) for name in self._config.reward_config.scales}
        obs = self._get_obs(data, info)
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.zeros(()),
            done=jp.zeros(()),
            metrics=metrics,
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if action.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},), got {action.shape}")
        action = jp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = jp.clip(action, -1.0, 1.0)
        desired_target = jp.clip(
            self._stand + action * self._config.action_scale,
            self._joint_low,
            self._joint_high,
        )
        target_buffer = jp.concatenate(
            [state.info["target_buffer"][1:], desired_target[None, :]], axis=0
        )
        delayed_index = int(self._config.maximum_action_delay_steps) - state.info[
            "action_delay_steps"
        ]
        delayed_target = target_buffer[delayed_index]
        previous_target = state.info["commanded_joints"]
        alpha = 1.0 - jp.exp(-self.dt / state.info["servo_time_constant"])
        lagged_target = previous_target + alpha * (delayed_target - previous_target)
        maximum_delta = self._config.servo_max_speed_rad_s * self.dt
        target = jp.clip(
            previous_target
            + jp.clip(lagged_target - previous_target, -maximum_delta, maximum_delta),
            self._joint_low,
            self._joint_high,
        )
        data = mjx_env.step(self.mjx_model, state.data, target, self.n_substeps)
        done = self._termination(data)
        reward_terms = self._reward_terms(data, action, state.info, done)
        scaled_terms = {
            name: value * self._config.reward_config.scales[name]
            for name, value in reward_terms.items()
        }
        reward = sum(scaled_terms.values()) * self.dt
        reward = jp.nan_to_num(
            reward,
            nan=self._config.reward_config.scales.termination * self.dt,
            posinf=0.0,
            neginf=self._config.reward_config.scales.termination * self.dt,
        )

        state.info["steps_until_command"] -= 1
        state.info["rng"], command_rng, time_rng = jax.random.split(state.info["rng"], 3)
        resample = state.info["steps_until_command"] <= 0
        state.info["command"] = jp.where(
            resample,
            self._sample_command(command_rng),
            state.info["command"],
        )
        state.info["steps_until_command"] = jp.where(
            resample,
            self._sample_command_steps(time_rng),
            state.info["steps_until_command"],
        )
        state.info["target_buffer"] = jp.where(
            done,
            jp.tile(
                self._stand[None, :], (int(self._config.maximum_action_delay_steps) + 1, 1)
            ),
            target_buffer,
        )
        state.info["commanded_joints"] = jp.where(done, self._stand, target)
        state.info["last_action"] = jp.where(done, jp.zeros_like(action), action)
        obs = self._get_obs(data, state.info)
        for name, value in scaled_terms.items():
            state.metrics[f"reward/{name}"] = value
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(reward.dtype),
        )

    def _actor_observation(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        up = self._up_vector(data)
        gyro = self._sensor(data, "imu_gyro")
        accelerometer = self._sensor(data, "imu_accel") / 9.81

        info["rng"], up_rng, gyro_rng, accel_rng = jax.random.split(info["rng"], 4)
        noise = self._config.noise_config
        up += jax.random.uniform(
            up_rng,
            up.shape,
            minval=-noise.gravity * noise.level,
            maxval=noise.gravity * noise.level,
        )
        gyro += jax.random.uniform(
            gyro_rng,
            gyro.shape,
            minval=-noise.gyro * noise.level,
            maxval=noise.gyro * noise.level,
        )
        accelerometer += jax.random.uniform(
            accel_rng,
            accelerometer.shape,
            minval=-noise.accelerometer * noise.level,
            maxval=noise.accelerometer * noise.level,
        )
        commanded = info["commanded_joints"] - self._stand
        actor = jp.hstack(
            [
                up,
                accelerometer,
                gyro,
                info["command"] / self._command_scale,
                commanded,
            ]
        )
        if actor.shape != (ACTOR_OBSERVATION_SIZE,):
            raise RuntimeError(f"Actor observation drifted to {actor.shape}")
        return actor

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        actor = self._actor_observation(data, info)
        privileged = jp.hstack(
            [
                actor,
                data.qpos,
                data.qvel,
                data.actuator_force,
                self._contacts(data),
            ]
        )
        if privileged.shape != (PRIVILEGED_OBSERVATION_SIZE,):
            raise RuntimeError(f"Privileged observation drifted to {privileged.shape}")
        return {"state": actor, "privileged_state": privileged}

    def _reward_terms(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        command = info["command"]
        local_velocity = self._sensor(data, "imu_velocity")
        gyro = self._sensor(data, "imu_gyro")
        true_joints = data.qpos[self._joint_qpos]
        joint_velocity = data.qvel[self._joint_dof]
        linear_error = jp.sum(jp.square(local_velocity[:2] - command[:2]))
        yaw_error = jp.square(gyro[2] - command[2])
        pose_error = jp.mean(jp.square(true_joints - self._stand))
        power = jp.sum(jp.abs(data.actuator_force * joint_velocity))
        action_rate = jp.mean(jp.square(action - info["last_action"]))
        upright = jp.clip(self._up_vector(data)[2], 0.0, 1.0)
        return {
            "tracking_linear_velocity": jp.exp(
                -linear_error / self._config.reward_config.tracking_sigma**2
            ),
            "tracking_yaw_velocity": jp.exp(
                -yaw_error / self._config.reward_config.yaw_tracking_sigma**2
            ),
            "pose": jp.exp(-pose_error / self._config.reward_config.pose_sigma**2),
            "upright": upright,
            "energy": power,
            "action_rate": action_rate,
            "termination": done,
        }

    def _termination(self, data: mjx.Data) -> jax.Array:
        height = data.qpos[self._root_qpos + 2]
        upright = self._up_vector(data)[2]
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        return (
            (height < self._config.termination_height)
            | (upright < self._config.termination_upright)
            | ~finite
        )
