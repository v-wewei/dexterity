"""Tests for smooth_action."""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf

from dexterity.effectors import hand_effector
from dexterity.effectors.wrappers import previous_action
from dexterity.effectors.wrappers import smooth_action
from dexterity.models.hands import shadow_hand_e


class ExponentialSmootherTest(absltest.TestCase):
    def test_raises_value_error_if_alpha_is_not_between_0_and_1(self) -> None:
        with self.assertRaises(ValueError):
            smooth_action.ExponentialSmoother(alpha=-0.1)
        with self.assertRaises(ValueError):
            smooth_action.ExponentialSmoother(alpha=1.1)

    def test_raises_value_error_if_no_update(self) -> None:
        with self.assertRaises(ValueError):
            smooth_action.ExponentialSmoother(alpha=0.0).smoothed_value

    def test_smoothed_value(self) -> None:
        # With a smoothing factor of 0.0, there is no smoothing, i.e., the smoothed
        # value should be equal to the most recent value.
        ema = smooth_action.ExponentialSmoother(alpha=0.0)
        ema.update(np.asarray([5]))
        ema.update(np.asarray([5.8]))
        self.assertEqual(ema.smoothed_value, np.asarray([5.8]))
        ema.update(np.asarray([100]))
        ema.update(np.asarray([-20]))
        self.assertEqual(ema.smoothed_value, np.asarray([-20]))

        # With a smoothing factor of 1.0, the most recent value is ignored and the
        # smoothed value will always be equal to the first value.
        ema = smooth_action.ExponentialSmoother(alpha=1.0)
        ema.update(np.asarray([5]))
        ema.update(np.asarray([5.8]))
        self.assertEqual(ema.smoothed_value, np.asarray([5]))
        ema.update(np.asarray([100]))
        ema.update(np.asarray([-20]))
        self.assertEqual(ema.smoothed_value, np.asarray([5]))


class SmoothActionTest(parameterized.TestCase):
    def test_initialize_episode(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()

        effector = hand_effector.HandEffector(hand=hand, hand_name=hand.name)
        smoothed_effector = smooth_action.SmoothAction(effector, alpha=0.5)

        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        random_state = np.random.RandomState(0)

        # At initialization, the ema variable should be initialized.
        smoothed_effector.initialize_episode(physics, random_state)
        self.assertIsNone(smoothed_effector._ema._value)

    @parameterized.parameters(0.0, 0.9, 1.0)
    def test_smoothing(self, alpha: float) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()

        effector = hand_effector.HandEffector(hand=hand, hand_name=hand.name)

        # Note: The wrapper order is important! We want to cache the smoothed actions so
        # the previous action wrapper must be called before the smooth action wrapper.
        prev_effector = previous_action.PreviousAction(effector)
        smoothed_effector = smooth_action.SmoothAction(prev_effector, alpha=alpha)

        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        random_state = np.random.RandomState(0)
        smoothed_effector.initialize_episode(physics, random_state)

        action_spec = smoothed_effector.action_spec(physics)

        def generate_random_control() -> np.ndarray:
            rand_ctrl = random_state.uniform(action_spec.minimum, action_spec.maximum)
            return rand_ctrl.astype(action_spec.dtype)

        # No smoothing should occur until the second update.
        rand_ctrl_0 = generate_random_control()
        smoothed_effector.set_control(physics, rand_ctrl_0)
        np.testing.assert_equal(smoothed_effector.previous_action, rand_ctrl_0)

        # At this point, the action that was applied should be a smoothed version of
        # the previous and current timesteps.
        rand_ctrl_1 = generate_random_control()
        smoothed_effector.set_control(physics, rand_ctrl_1)
        np.testing.assert_equal(
            smoothed_effector.previous_action,
            (1 - alpha) * rand_ctrl_1 + alpha * rand_ctrl_0,
        )
