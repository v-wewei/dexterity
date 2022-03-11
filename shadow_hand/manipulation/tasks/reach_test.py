"""Tests reach.py."""

import numpy as np
from absl.testing import absltest

from shadow_hand import manipulation


class ReachTaskTest(absltest.TestCase):
    def test_dense_reward(self) -> None:
        env = manipulation.load(environment_name="reach_state_dense")
        env.reset()

        zero_action = np.zeros_like(env.physics.data.ctrl)
        timestep = env.step(zero_action)

        target_positions = timestep.observation["target_positions"]
        fingertip_positions = timestep.observation["shadow_hand_e/fingertip_positions"]
        expected_reward = -1.0 * np.linalg.norm(target_positions - fingertip_positions)
        np.testing.assert_equal(expected_reward, timestep.reward)


if __name__ == "__main__":
    absltest.main()
