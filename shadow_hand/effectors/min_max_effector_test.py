"""Tests for min_max_effector."""

import numpy as np
from absl.testing import absltest
from dm_control import mjcf
from dm_env import specs

from shadow_hand import effector
from shadow_hand.effectors import min_max_effector


class DummyEffector(effector.Effector):
    """An effector that stores the most recent command for testing purposes."""

    def __init__(self, dofs: int) -> None:
        self._previous_action = np.empty(dofs)

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=self._previous_action.shape,
            dtype=self._previous_action.dtype,
            minimum=-1.0,
            maximum=1.0,
        )

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        self._previous_action[:] = command

    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        pass

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        pass

    @property
    def prefix(self) -> str:
        return "test"

    @property
    def previous_action(self) -> np.ndarray:
        return self._previous_action


class MinMaxEffectorTest(absltest.TestCase):
    def test_min_max_effector_sends_correct_command(self) -> None:
        base_effector = DummyEffector(3)
        min_action = np.array([-0.9, -0.5, -0.2])
        max_action = np.array([0.2, 0.5, 0.8])
        test_effector = min_max_effector.MinMaxEffector(
            base_effector=base_effector,
            min_action=min_action,
            max_action=max_action,
        )
        sent_command = np.array([-0.8, 0.0, 0.3])
        expected_command = np.array([-0.9, -0.5, 0.8])
        test_effector.set_control(None, sent_command)
        np.testing.assert_allclose(
            expected_command,
            base_effector.previous_action,
        )

    def test_min_max_effector_default_action_spec(self) -> None:
        base_effector = DummyEffector(3)
        test_effector = min_max_effector.MinMaxEffector(base_effector=base_effector)
        sent_command = np.array([-0.8, 0.0, 0.3])
        expected_command = np.array([-1.0, -1.0, 1.0])
        test_effector.set_control(None, sent_command)
        np.testing.assert_allclose(
            expected_command,
            base_effector.previous_action,
        )


if __name__ == "__main__":
    absltest.main()
