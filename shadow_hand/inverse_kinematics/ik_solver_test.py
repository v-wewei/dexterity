"""Tests for ik_solver."""

import numpy as np
from absl.testing import absltest

from shadow_hand.inverse_kinematics import ik_solver
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts


class IKSolverTest(absltest.TestCase):
    def test_return_none_when_passing_impossible_target(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        target_positions = {
            consts.Components.TH: np.array([5.0, 5.0, 5.0]),
        }
        solver = ik_solver.IKSolver(
            hand.mjcf_model, fingers=tuple(target_positions.keys())
        )
        qpos_sol = solver.solve(target_positions)
        self.assertIsNone(qpos_sol)


if __name__ == "__main__":
    absltest.main()
