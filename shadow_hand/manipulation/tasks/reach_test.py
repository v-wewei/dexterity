"""Tests for reach.py."""

import numpy as np
from absl.testing import absltest
from dm_control import composer

from shadow_hand.manipulation.shared import observations
from shadow_hand.manipulation.shared import rewards
from shadow_hand.manipulation.tasks import reach_task
from shadow_hand.manipulation.tasks.reach import _DISTANCE_TO_TARGET_THRESHOLD


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

        rand_action = random_state.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape,
        )
        timestep = env.step(rand_action)

        target_positions = timestep.observation["target_positions"]
        fingertip_positions = timestep.observation["shadow_hand_e/fingertip_positions"]
        distance = np.linalg.norm(
            target_positions.reshape(-1, 3) - fingertip_positions.reshape(-1, 3),
            axis=1,
        )
        expected_reward = np.mean(
            np.where(
                distance <= _DISTANCE_TO_TARGET_THRESHOLD,
                1.0,
                [1.0 - rewards.tanh_squared(d, margin=0.1) for d in distance],
            )
        )
        np.testing.assert_equal(timestep.reward, expected_reward)

    def test_sparse_reward(self) -> None:
        task = reach_task(
            observations.ObservationSet.STATE_ONLY,
            use_dense_reward=False,
            visualize_reward=False,
        )

        random_state = np.random.RandomState(12345)
        env = composer.Environment(task, random_state=random_state)

        env.reset()
        timestep = env.step(np.zeros(env.action_spec().shape))
        self.assertEqual(timestep.reward, 0.0)

        assert env.task._fingertips_initializer.qpos is not None
        qpos_sol = env.task._fingertips_initializer.qpos.copy()
        ctrl_sol = env.task.hand.joint_positions_to_control(qpos_sol)
        env.physics.bind(env.task.hand.joints).qpos = qpos_sol
        timestep = env.step(ctrl_sol)
        target_positions = timestep.observation["target_positions"]
        fingertip_positions = timestep.observation["shadow_hand_e/fingertip_positions"]
        distance = np.linalg.norm(
            target_positions.reshape(-1, 3) - fingertip_positions.reshape(-1, 3),
            axis=1,
        )
        expected_reward = np.mean(
            np.where(distance <= _DISTANCE_TO_TARGET_THRESHOLD, 1.0, 0.0)
        )
        np.testing.assert_equal(timestep.reward, expected_reward)


if __name__ == "__main__":
    absltest.main()
