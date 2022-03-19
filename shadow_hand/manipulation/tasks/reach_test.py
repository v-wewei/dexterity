"""Tests for reach.py."""

import numpy as np
from absl.testing import absltest
from dm_control import composer

from shadow_hand.manipulation.shared import observations
from shadow_hand.manipulation.tasks import reach_task


class ReachTaskTest(absltest.TestCase):
    def test_dense_reward(self) -> None:
        task = reach_task(
            observations.ObservationSet.STATE_ONLY,
            use_dense_reward=True,
            visualize_reward=False,
        )

        random_state = np.random.RandomState(12345)
        env = composer.Environment(task, random_state=random_state)
        action_spec = env.action_spec()
        timestep = env.reset()
        self.assertIsNone(timestep.reward)

        rand_action = np.random.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape,
        )
        timestep = env.step(rand_action)

        target_positions = timestep.observation["target_positions"]
        fingertip_positions = timestep.observation["shadow_hand_e/fingertip_positions"]
        expected_reward = -1.0 * np.linalg.norm(target_positions - fingertip_positions)
        np.testing.assert_equal(expected_reward, timestep.reward)

        env.reset()

        assert env.task._fingertips_initializer.qpos is not None
        env.physics.bind(
            env.task._hand.joints
        ).qpos = env.task._fingertips_initializer.qpos
        env.physics.step()
        env.task.after_step(env.physics, random_state)
        actual_reward = env.task.get_reward(env.physics)
        np.testing.assert_almost_equal(actual_reward, 0.0)


if __name__ == "__main__":
    absltest.main()
