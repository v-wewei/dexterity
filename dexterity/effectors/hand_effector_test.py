"""Tests for hand_effector."""

import numpy as np
from absl.testing import absltest
from dm_control import mjcf

from dexterity.effectors import hand_effector
from dexterity.models.hands import shadow_hand_e


class HandEffectorTest(absltest.TestCase):
    def test_set_control(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        effector = hand_effector.HandEffector(hand=hand, hand_name=hand.name)
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        action_spec = effector.action_spec(physics)
        rand_ctrl = np.random.uniform(*physics.bind(hand.actuators).ctrlrange.T)
        rand_ctrl = rand_ctrl.astype(action_spec.dtype)
        effector.set_control(physics, rand_ctrl)
        np.testing.assert_allclose(physics.bind(hand.actuators).ctrl, rand_ctrl)


if __name__ == "__main__":
    absltest.main()
