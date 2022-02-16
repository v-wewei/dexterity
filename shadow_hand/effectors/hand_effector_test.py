"""Tests for min_max_effector."""

import numpy as np
from absl.testing import absltest
from dm_control import mjcf

from shadow_hand.effectors import hand_effector
from shadow_hand.models.hands import shadow_hand_e


class HandEffectorTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self._hand = shadow_hand_e.ShadowHandSeriesE()
        self._physics = mjcf.Physics.from_mjcf_model(self._hand.mjcf_model)

    def test_set_control(self) -> None:
        effector = hand_effector.HandEffector(self._hand, "test_shadow_hand")
        control = np.ones(len(self._hand.actuators), dtype=np.float32) * 0.01
        effector.set_control(self._physics, control)
        np.testing.assert_allclose(
            self._physics.bind(self._hand.actuators).ctrl,
            control,
        )


if __name__ == "__main__":
    absltest.main()
