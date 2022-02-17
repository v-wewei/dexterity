"""Tests for ik_solver."""


import numpy as np
from absl.testing import absltest

from shadow_hand.ik import ik_solver
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts

# from dm_control import mjcf


_SEED = 0
_NUM_TESTS = 5
_LINEAR_TOL = 1e-4


class IKSolverTest(absltest.TestCase):
    def test_return_none_when_passing_impossible_target(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        solver = ik_solver.IKSolver(hand.mjcf_model)
        target_positions = {
            consts.Components.TH: np.array([5.0, 5.0, 5.0]),
        }
        qpos_sol = solver.solve(target_positions)
        self.assertIsNone(qpos_sol)

    # def test_solve_from_random_joint_config(self) -> None:
    #     np.random.seed(_SEED)
    #     rng = np.random.RandomState(_SEED)

    #     hand = shadow_hand_e.ShadowHandSeriesE()
    #     solver = ik_solver.IKSolver(hand.mjcf_model)

    #     for _ in range(_NUM_TESTS):
    #         # Sample a random joint configuration.
    #         physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
    #         joint_binding = physics.bind(hand.joints)
    #         qpos_expected = np.random.uniform(
    #             joint_binding.range[:, 0],
    #             joint_binding.range[:, 1],
    #         )
    #         qpos_expected[:2] = 0.0  # Disable wrist movement.

    #         physics.bind(hand.joints).qpos = qpos_expected
    #         target_positions = {}
    #         for fingertip_site in hand._fingertip_sites:
    #             target_positions[] = physics.bind(
    #                 fingertip_site
    #             ).xpos.copy()


if __name__ == "__main__":
    absltest.main()
