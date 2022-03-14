"""Tests for hand_effector."""

import numpy as np
from absl.testing import absltest
from dm_control import mjcf

from shadow_hand.effectors import hand_effector
from shadow_hand.models.hands import shadow_hand_e


class HandEffectorTest(absltest.TestCase):
    def test_set_control(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        effector = hand_effector.HandEffector(hand=hand, hand_name=hand.name)
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        rand_ctrl = np.random.uniform(
            low=physics.bind(hand.actuators).ctrlrange[:, 0],
            high=physics.bind(hand.actuators).ctrlrange[:, 1],
        )
        effector.set_control(physics, rand_ctrl)
        np.testing.assert_allclose(physics.bind(hand.actuators).ctrl, rand_ctrl)


if __name__ == "__main__":
    absltest.main()
