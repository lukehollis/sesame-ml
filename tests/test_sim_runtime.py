from __future__ import annotations

from sesame_ml.constants import STAND_ANGLES_RAD
from sesame_ml.envs import SesameEnv, SesameEnvConfig, Task
from sesame_ml.runtime import SimulatedRobotRuntime
from sesame_ml.transport import ActionSample, ControlState


def test_sim_runtime_exposes_deployment_modalities_and_accepts_physical_targets() -> None:
    env = SesameEnv(
        task=Task.NAVIGATION,
        config=SesameEnvConfig(
            task=Task.NAVIGATION,
            observation_mode="pixels",
            camera_width=80,
            camera_height=60,
            domain_randomization=False,
        ),
    )
    runtime = SimulatedRobotRuntime(env, seed=5)
    observation = runtime.observation_source(3, 1000)
    assert observation.sequence == 3
    assert observation.imu_quaternion is not None
    assert observation.imu_gyro is not None
    assert observation.image_jpeg is not None and observation.image_jpeg.startswith(b"\xff\xd8")
    runtime.action_sink(
        ActionSample(
            target=tuple(float(value) for value in STAND_ANGLES_RAD),
            state=ControlState.ACTIVE,
            reason="test",
        )
    )
    assert runtime.metrics.steps == 1
    runtime.close()
