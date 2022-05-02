"""Tests for ik_solver."""

import numpy as np
from absl.testing import absltest
from dm_control import mjcf
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics

from dexterity.inverse_kinematics import ik_solver
from dexterity.models import hands

_SEED = 12345
_LINEAR_TOL = 1e-3
_NUM_SOLVES = 50


class IKSolverTest(absltest.TestCase):
    def test_raises_value_error_if_target_wrong_shape(self) -> None:
        hand = hands.ShadowHandSeriesE()
        target_positions = np.full(shape=(1, 3), fill_value=10.0)
        solver = ik_solver.IKSolver(hand)
        with self.assertRaises(ValueError):
            solver.solve(target_positions)

    def test_return_none_when_passing_impossible_target(self) -> None:
        hand = hands.ShadowHandSeriesE()
        target_positions = np.full(shape=(5, 3), fill_value=10.0)
        solver = ik_solver.IKSolver(hand)
        qpos_sol = solver.solve(target_positions)
        self.assertIsNone(qpos_sol)

    def test_with_feasible_targets(self) -> None:
        np.random.seed(_SEED)
        random_state = np.random.RandomState(_SEED)

        hand = hands.AdroitHand()  # Use a fully-actuated hand.
        solver = ik_solver.IKSolver(hand)

        for _ in range(_NUM_SOLVES):
            targets = _sample_reachable_targets(solver._physics, hand, random_state)
            ref_poses = [geometry.Pose(target) for target in targets]

            # Check that we can solve all finger targets.
            qpos_sol = solver.solve(
                targets,
                linear_tol=_LINEAR_TOL,
                early_stop=True,
                stop_on_first_successful_attempt=True,
            )
            self.assertIsNotNone(qpos_sol)

            # Ensure the solutions are within the joint limits.
            min_range = solver._all_joints_binding.range[:, 0]
            max_range = solver._all_joints_binding.range[:, 1]
            assert qpos_sol is not None  # Appease mypy.
            np.testing.assert_array_compare(np.less_equal, qpos_sol, max_range)
            np.testing.assert_array_compare(np.greater_equal, qpos_sol, min_range)

            # Check max linear error is satisfied for all fingers.
            geometry_physics = mujoco_physics.wrap(solver._physics)
            solver._all_joints_binding.qpos[:] = qpos_sol
            for i, ik_site in enumerate(hand.fingertip_sites):
                end_pose = geometry_physics.world_pose(ik_site)
                linear_error = np.linalg.norm(end_pose.position - ref_poses[i].position)
                self.assertLessEqual(linear_error, _LINEAR_TOL)


def _sample_reachable_targets(
    physics: mjcf.Physics,
    hand: hands.DexterousHand,
    random_state: np.random.RandomState,
) -> np.ndarray:
    # Save the initial joint configuration.
    qpos_initial = physics.bind(hand.joints).qpos.copy()

    # Sample a collision-free configuration, set the hand to it, and compute fingertip
    # poses with forward kinematics.
    qpos = hand.sample_collision_free_joint_angles(physics, random_state)
    physics.bind(hand.joints).qpos[:] = qpos
    # Note: When we bind to the joints with the physics object and modify the value of
    # qpos, the value of xpos is automatically recalculated, so we don't have to call
    # physics.forward() explicitly.
    target_positions = physics.bind(hand.fingertip_sites).xpos.copy()

    # Restore the initial joint configuration.
    physics.bind(hand.joints).qpos[:] = qpos_initial

    return target_positions


if __name__ == "__main__":
    absltest.main()
