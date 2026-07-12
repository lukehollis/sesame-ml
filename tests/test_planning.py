from __future__ import annotations

import numpy as np

from sesame_ml.envs import SesameEnv, Task
from sesame_ml.planning import (
    AxisAlignedObstacle,
    CEMConfig,
    MuJoCoCEMPlanner,
    OccupancyGridPlanner,
)


def test_waypoint_planner_routes_around_inflated_obstacle() -> None:
    obstacle = AxisAlignedObstacle((0.5, 0.0), (0.10, 0.18))
    planner = OccupancyGridPlanner(resolution_m=0.04, robot_radius_m=0.08)
    path = planner.plan((0, 0), (1, 0), [obstacle])
    assert np.allclose(path[0], [0, 0])
    assert np.allclose(path[-1], [1, 0])
    assert len(path) >= 3
    assert all(not obstacle.contains(point, 0.08) for point in path)


def test_cem_planner_returns_bounded_chunk_without_mutating_live_state() -> None:
    env = SesameEnv(task=Task.LOCOMOTION, domain_randomization=False)
    env.reset(seed=3)
    before_qpos = env.data.qpos.copy()
    before_time = env.data.time
    planner = MuJoCoCEMPlanner(
        env,
        CEMConfig(horizon=3, population=5, elites=2, iterations=1, chunk_steps=2, seed=4),
    )
    plan = planner.plan(command=np.zeros(3))
    assert plan.actions.shape == (2, 8)
    assert np.all(np.abs(plan.actions) <= 1)
    assert np.isfinite(plan.predicted_return)
    assert np.array_equal(env.data.qpos, before_qpos)
    assert env.data.time == before_time
    env.close()
