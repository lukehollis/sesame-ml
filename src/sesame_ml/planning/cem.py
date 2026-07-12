"""Receding-horizon cross-entropy control using the real Sesame MuJoCo dynamics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sesame_ml.constants import SERVO_MAX_SPEED_RAD_S, STAND_ANGLES_RAD, clip_servo_targets
from sesame_ml.envs.core import SesameEnv, Task
from sesame_ml.model import joint_dof_addresses


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 18
    population: int = 192
    elites: int = 24
    iterations: int = 4
    initial_std: float = 0.55
    minimum_std: float = 0.06
    momentum: float = 0.15
    chunk_steps: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least two control steps")
        if not 1 <= self.elites < self.population:
            raise ValueError("elites must be positive and smaller than population")
        if self.iterations < 1 or not 1 <= self.chunk_steps <= self.horizon:
            raise ValueError("invalid iterations or chunk_steps")


@dataclass(frozen=True)
class CEMPlan:
    actions: np.ndarray
    predicted_return: float
    population_std: float
    elapsed_s: float


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / max(float(np.linalg.norm(quat)), 1e-12)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class MuJoCoCEMPlanner:
    """Sample-based MPC that plans directly through the calibrated MuJoCo model.

    The planner is intentionally independent of Gym's reward implementation: it clones the
    live simulator state, predicts candidate action sequences without modifying the live
    episode, and returns a short chunk for receding-horizon execution. On real hardware the
    same chunk contract is sent over Wi-Fi while state estimation supplies the initial state.
    """

    def __init__(self, env: SesameEnv, config: CEMConfig | None = None) -> None:
        self.env = env
        self.config = config or CEMConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._data = mujoco.MjData(env.model)
        self._dof_adr = joint_dof_addresses(env.model)
        root_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self._root_qpos = int(env.model.jnt_qposadr[root_id])
        self._root_dof = int(env.model.jnt_dofadr[root_id])
        self._obstacle_body_ids = {
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_0"),
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_1"),
        }
        self._mean = np.zeros((self.config.horizon, 8), dtype=np.float64)

    def reset(self) -> None:
        self._mean[:] = 0

    def _copy_live_state(self) -> None:
        source, target = self.env.data, self._data
        for name in (
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "mocap_pos",
            "mocap_quat",
            "qacc_warmstart",
            "xfrc_applied",
        ):
            source_value = getattr(source, name)
            target_value = getattr(target, name)
            if source_value.size:
                target_value[:] = source_value
        target.time = source.time
        mujoco.mj_forward(self.env.model, target)

    def _snapshot(self) -> dict[str, np.ndarray | float]:
        return {
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "act": self._data.act.copy(),
            "ctrl": self._data.ctrl.copy(),
            "mocap_pos": self._data.mocap_pos.copy(),
            "mocap_quat": self._data.mocap_quat.copy(),
            "time": float(self._data.time),
        }

    def _restore(self, state: dict[str, np.ndarray | float]) -> None:
        for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
            value = state[name]
            if isinstance(value, np.ndarray) and value.size:
                getattr(self._data, name)[:] = value
        self._data.time = float(state["time"])
        mujoco.mj_forward(self.env.model, self._data)

    def _has_obstacle_contact(self) -> bool:
        for contact in self._data.contact[: self._data.ncon]:
            body_a = int(self.env.model.geom_bodyid[contact.geom1])
            body_b = int(self.env.model.geom_bodyid[contact.geom2])
            if (body_a in self._obstacle_body_ids) ^ (body_b in self._obstacle_body_ids):
                return True
        return False

    def _rollout(
        self,
        actions: np.ndarray,
        snapshot: dict[str, np.ndarray | float],
        command: np.ndarray,
        goal: np.ndarray | None,
    ) -> float:
        self._restore(snapshot)
        filtered = self.env.servo_targets
        previous_action = self.env._last_action  # Planner needs the live rate-cost boundary.
        initial_goal_distance = 0.0
        if goal is not None:
            initial_goal_distance = float(
                np.linalg.norm(goal[:2] - self._data.qpos[self._root_qpos : self._root_qpos + 2])
            )
        total = 0.0
        discount = 1.0
        alpha = 1.0 - math.exp(
            -self.env.config.control_dt / self.env._domain.servo_time_constant_s
        )
        max_delta = SERVO_MAX_SPEED_RAD_S * self.env.config.control_dt

        for action in actions:
            desired = clip_servo_targets(
                STAND_ANGLES_RAD + self.env.config.action_scale_rad * np.clip(action, -1, 1)
            )
            lagged = filtered + alpha * (desired - filtered)
            filtered += np.clip(lagged - filtered, -max_delta, max_delta)
            self._data.ctrl[:] = filtered
            for _ in range(self.env.frame_skip):
                mujoco.mj_step(self.env.model, self._data)

            qpos = self._data.qpos
            rotation = _quat_to_matrix(qpos[self._root_qpos + 3 : self._root_qpos + 7])
            up = float((rotation.T @ np.asarray([0.0, 0.0, 1.0]))[2])
            height = float(qpos[self._root_qpos + 2])
            world_velocity = self._data.qvel[self._root_dof : self._root_dof + 3]
            local_velocity = rotation.T @ world_velocity
            angular_velocity = self._data.qvel[self._root_dof + 3 : self._root_dof + 6]
            linear_error = float(np.sum((local_velocity[:2] - command[:2]) ** 2))
            yaw_error = float((angular_velocity[2] - command[2]) ** 2)
            energy = float(
                np.sum(np.abs(self._data.actuator_force * self._data.qvel[self._dof_adr]))
            )
            action_rate = float(np.mean((action - previous_action) ** 2))
            stage = (
                1.20 * math.exp(-linear_error / 0.10**2)
                + 0.55 * math.exp(-yaw_error / 0.55**2)
                + 0.35 * max(0.0, up)
                + 0.12 * math.exp(-((height - 0.054) / 0.018) ** 2)
                - 0.003 * energy
                - 0.035 * action_rate
            )
            if goal is not None:
                distance = float(
                    np.linalg.norm(goal[:2] - qpos[self._root_qpos : self._root_qpos + 2])
                )
                stage += 2.0 * (initial_goal_distance - distance) / self.config.horizon
                if self._has_obstacle_contact():
                    stage -= 0.4
            total += discount * stage
            discount *= 0.985
            previous_action = action
            if height < 0.024 or up < 0.15 or not np.all(np.isfinite(qpos)):
                total -= 25.0
                break
        return total

    def plan(
        self,
        *,
        command: np.ndarray | None = None,
        goal: np.ndarray | None = None,
    ) -> CEMPlan:
        """Plan from the environment's current state and return a safe executable chunk."""

        started = time.perf_counter()
        self._copy_live_state()
        snapshot = self._snapshot()
        command_value = (
            self.env.command if command is None else np.asarray(command, dtype=np.float64)
        )
        if command_value.shape != (3,):
            raise ValueError("command must have shape (3,)")
        if goal is None and self.env.task == Task.NAVIGATION:
            goal = self.env.goal
        if goal is not None:
            goal = np.asarray(goal, dtype=np.float64)
            if goal.shape != (3,):
                raise ValueError("goal must have shape (3,)")

        mean = self._mean.copy()
        std = np.full_like(mean, self.config.initial_std)
        best_return = -np.inf
        best_actions = mean.copy()
        for _ in range(self.config.iterations):
            noise = self._rng.normal(size=(self.config.population, self.config.horizon, 8))
            # Low-pass sampling produces servo-feasible sequences and a substantially better
            # optimizer signal than independent white-noise commands.
            noise[:, 1:] = 0.68 * noise[:, :-1] + 0.32 * noise[:, 1:]
            candidates = np.clip(mean[None, :, :] + std[None, :, :] * noise, -1, 1)
            candidates[0] = np.clip(mean, -1, 1)
            returns = np.asarray(
                [
                    self._rollout(candidate, snapshot, command_value, goal)
                    for candidate in candidates
                ]
            )
            elite_ids = np.argpartition(returns, -self.config.elites)[-self.config.elites :]
            elites = candidates[elite_ids]
            elite_mean = elites.mean(axis=0)
            elite_std = np.maximum(elites.std(axis=0), self.config.minimum_std)
            mean = self.config.momentum * mean + (1 - self.config.momentum) * elite_mean
            std = self.config.momentum * std + (1 - self.config.momentum) * elite_std
            iteration_best = int(np.argmax(returns))
            if returns[iteration_best] > best_return:
                best_return = float(returns[iteration_best])
                best_actions = candidates[iteration_best].copy()

        # Warm start the next online solve by shifting the optimized distribution.
        selected = mean if np.isfinite(mean).all() else best_actions
        shift = self.config.chunk_steps
        self._mean[:-shift] = selected[shift:]
        self._mean[-shift:] = selected[-1]
        return CEMPlan(
            actions=np.clip(selected[:shift], -1, 1).astype(np.float32),
            predicted_return=best_return,
            population_std=float(std.mean()),
            elapsed_s=time.perf_counter() - started,
        )
