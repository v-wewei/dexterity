"""Tests for reach.py."""

import numpy as np
from absl.testing import absltest
from dm_control import composer

from shadow_hand.manipulation.shared import observations
from shadow_hand.manipulation.tasks import reach_task
from shadow_hand.manipulation.tasks.reach import _REWARD_BONUS


class ReachTaskTest(absltest.TestCase):
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
        while True:
            timestep = env.step(ctrl_sol)
            if env.task.total_solves > 0:
                break
        expected_reward = 1 + _REWARD_BONUS
        np.testing.assert_equal(timestep.reward, expected_reward)


if __name__ == "__main__":
    absltest.main()
