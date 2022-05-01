"""Tests for previous_action."""

import numpy as np
from absl.testing import absltest
from dm_control import mjcf

from dexterity.effectors import hand_effector
from dexterity.effectors.wrappers import previous_action
from dexterity.models.hands import shadow_hand_e


class PreviousActionTest(absltest.TestCase):
    def test_previous_action_wrapper(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        effector = previous_action.PreviousAction(
            effector=hand_effector.HandEffector(hand=hand, hand_name=hand.name)
        )

        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        random_state = np.random.RandomState(0)

        self.assertIs(effector.previous_action, None)
        effector.initialize_episode(physics, random_state)
        self.assertIsNot(effector.previous_action, None)

        action_spec = effector.action_spec(physics)
        rand_ctrl = np.random.uniform(action_spec.minimum, action_spec.maximum)
        rand_ctrl = rand_ctrl.astype(action_spec.dtype)
        effector.set_control(physics, rand_ctrl)

        assert effector.previous_action is not None  # For mypy's sake.
        np.testing.assert_array_equal(effector.previous_action, rand_ctrl)
