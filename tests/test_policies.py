from __future__ import annotations

import numpy as np
import pytest

from sesame_ml.constants import STAND_ANGLES_RAD
from sesame_ml.policies import (
    ActionMode,
    CPGPolicy,
    FirmwareMotion,
    FirmwareSequenceConfig,
    FirmwareSequencePolicy,
    SB3Policy,
    StandPolicy,
    predict_action,
    proprioception,
    residual_action,
)


def observation(*, forward: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    state = np.zeros(43, dtype=np.float32)
    state[2] = 1.0
    state[33] = forward
    state[35] = yaw
    return state


def test_policy_boundary_enforces_residual_action_semantics() -> None:
    assert np.array_equal(residual_action(np.full(8, 2.0)), np.ones(8))
    with pytest.raises(ValueError, match="shape"):
        residual_action(np.zeros(7))
    with pytest.raises(ValueError, match="non-finite"):
        residual_action(np.full(8, np.nan))
    with pytest.raises(ValueError, match="outside"):
        residual_action(np.full(8, 1.1), clip=False)
    assert np.array_equal(proprioception({"proprio": observation()}), observation())


def test_stand_policy_is_zero_at_level_stand_and_damps_tilt() -> None:
    policy = StandPolicy()
    assert np.array_equal(policy.predict(observation()), np.zeros(8))
    tilted = observation()
    tilted[0] = 0.2
    action = policy.predict(tilted)
    assert action.shape == (8,)
    assert np.any(action != 0)
    assert np.all(np.abs(action) <= 1)


def test_cpg_reset_reproduces_seeded_trajectory() -> None:
    policy = CPGPolicy()
    policy.reset(seed=71)
    first = [policy.predict(observation(forward=0.8), deterministic=False) for _ in range(10)]
    policy.reset(seed=71)
    second = [policy.predict(observation(forward=0.8), deterministic=False) for _ in range(10)]
    assert np.allclose(first, second)
    assert all(action.shape == (8,) for action in first)
    assert all(np.all(np.abs(action) <= 1.0) for action in first)


def test_firmware_sequence_replays_absolute_keyframes_then_clips_residuals() -> None:
    policy = FirmwareSequencePolicy(
        FirmwareSequenceConfig(motion=FirmwareMotion.FORWARD, interpolate=False)
    )
    action = policy.predict(observation(forward=1.0))
    target_degrees = np.rad2deg(policy.servo_targets_rad)
    # Initial step from runWalkPose in firmware/movement-sequences.h.
    assert target_degrees[[1, 2, 5, 6]] == pytest.approx([100, 25, 135, 45])
    expected = np.clip((policy.servo_targets_rad - STAND_ANGLES_RAD) / 0.58, -1, 1)
    assert action == pytest.approx(expected)
    assert policy.evaluation_name == "firmware_sequence"
    assert np.any(np.abs(expected) == 1.0), "firmware excursions should document residual clipping"


def test_firmware_auto_selects_commanded_turn_and_stands_inside_deadband() -> None:
    policy = FirmwareSequencePolicy()
    policy.predict(observation(yaw=0.9))
    assert policy.motion is FirmwareMotion.TURN_LEFT
    policy.predict(observation(yaw=-0.9))
    assert policy.motion is FirmwareMotion.TURN_RIGHT
    action = policy.predict(observation())
    assert policy.motion is None
    assert np.array_equal(action, np.zeros(8))


def test_firmware_forward_prefix_is_not_reintroduced_when_cycle_wraps() -> None:
    policy = FirmwareSequencePolicy(
        FirmwareSequenceConfig(
            motion=FirmwareMotion.FORWARD,
            control_dt=0.1,
            frame_duration_s=0.1,
            interpolate=False,
        )
    )
    targets = []
    for _ in range(8):
        policy.predict(observation(forward=1.0))
        targets.append(np.rad2deg(policy.servo_targets_rad))
    # Frame seven is the first keyframe of the second loop.  R2/L1 must retain
    # frame six's 90/0 values, rather than jumping back to the one-time 100/25 prefix.
    assert targets[7][[1, 2]] == pytest.approx([90, 0])


class _DummySB3Model:
    class _Space:
        shape = (8,)

    action_space = _Space()

    def predict(self, observation, **kwargs):  # type: ignore[no-untyped-def]
        del observation, kwargs
        return np.full(8, 0.25, dtype=np.float32), None


def test_sb3_adapter_uses_policy_contract() -> None:
    policy = SB3Policy(_DummySB3Model())
    policy.reset(seed=1)
    assert policy.action_mode is ActionMode.RESIDUAL
    assert predict_action(policy, observation()) == pytest.approx(np.full(8, 0.25))
