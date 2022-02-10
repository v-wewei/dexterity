"""Differential inverse kinematics."""

import dataclasses
from typing import Optional, Sequence

import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings

from shadow_hand.hints import MjcfElement

mjlib = mjbindings.mjlib

# Integration timestep used when solving the IK.
_INTEGRATION_TIMESTEP_SEC = 1.0


@dataclasses.dataclass
class _Solution:
    """Return value of an IK solution."""

    qpos: np.ndarray
    linear_err: float


class DifferentialIK:
    """Differential inverse kinematics."""

    def __init__(
        self,
        model: mjcf.RootElement,
        controllable_joints: Sequence[MjcfElement],
        site_name: str,
    ) -> None:

        # self._physics = physics
        self._physics = mjcf.Physics.from_mjcf_model(model)
        self._controllable_joints = controllable_joints
        self._num_joints = len(self._controllable_joints)
        self._site_name = site_name

        # Default nullspace referrence is the mid-range of the joints.
        self._joints_binding = self._physics.bind(self._controllable_joints)
        self._nullspace_joint_position_reference = 0.5 * np.sum(
            self._joints_binding.range, axis=1
        )

    def solve(
        self,
        target_position: np.ndarray,
        linear_tol: float = 1e-3,
        max_steps: int = 100,
        early_stop: bool = False,
        num_attempts: int = 30,
        stop_on_first_successful_attempt: bool = False,
        inital_joint_configuration: Optional[np.ndarray] = None,
        nullspace_reference: Optional[np.ndarray] = None,
        regularization_weight: float = 1e-3,
    ) -> Optional[np.ndarray]:
        """Attempts to solve the IK problem."""

        if nullspace_reference is None:
            nullspace_reference = self._nullspace_joint_position_reference
        else:
            if len(nullspace_reference) != self._num_joints:
                raise ValueError(
                    f"Expected {self._num_joints} elements for the nullspace reference "
                    f"Got {len(nullspace_reference)}."
                )

        if inital_joint_configuration is not None:
            if len(inital_joint_configuration) != self._num_joints:
                raise ValueError(
                    f"Expected {self._num_joints} elements for the initial joint "
                    f"configuration. Got {len(inital_joint_configuration)}."
                )

        if inital_joint_configuration is None:
            inital_joint_configuration = np.zeros(self._num_joints)

        nullspace_jnt_qpos_min_err: float = np.inf
        sol_qpos: Optional[np.ndarray] = None
        success: bool = False

        # Each iteration of this loop attempts to solve the IK problem. If a solution
        # is found, it is compared to previous solutions.
        for attempt in range(num_attempts):
            # Use the user provided joint configuration for the first attempt.
            if attempt == 0:
                self._joints_binding.qpos[:] = inital_joint_configuration
            else:
                # Randomize the initial joint configuration so that the IK can find
                # different solutions.
                qpos_new = np.random.uniform(
                    self._joints_binding.range[:, 0],
                    self._joints_binding.range[:, 1],
                )
                self._joints_binding.qpos[:] = qpos_new

            sol = self._solve_ik(
                target_position,
                linear_tol,
                max_steps,
                early_stop,
                regularization_weight,
            )

            # Check if the attempt was successful. The solution is saved if the joints
            # are closer to the nullspace reference.
            if sol.linear_err < linear_tol:
                success = True
                nullspace_jnt_qpos_err = float(
                    np.linalg.norm(sol.qpos - nullspace_reference)
                )
                if nullspace_jnt_qpos_err < nullspace_jnt_qpos_min_err:
                    nullspace_jnt_qpos_min_err = nullspace_jnt_qpos_err
                    sol_qpos = sol.qpos

            if success and stop_on_first_successful_attempt:
                break

        if not success:
            print(f"{self.__class__.__name__} failed to find a solution.")

        return sol_qpos

    def _solve_ik(
        self,
        target_position: np.ndarray,
        linear_tol: float,
        max_steps: int,
        early_stop: bool,
        regularization_weight: float,
    ) -> _Solution:
        # Convert site name to index.
        site_id = self._physics.model.name2id(self._site_name, "site")

        # This is a view on the underlying MuJoCo buffer. `mj_fwdPosition`` will update
        # it in place, so we can avoid indexing overhead in the main loop.
        site_xpos = self._physics.named.data.site_xpos[self._site_name]

        linear_err: float = np.inf

        # Each iteration of this loop attempts to reduce the error between the site's
        # position and the target position.
        for _ in range(max_steps):

            qdot_sol = np.zeros(self._physics.model.nv)
            joint_vel = self._compute_joint_velocities(
                site_id,
                target_position,
                site_xpos,
                regularization_weight,
            )

            if joint_vel is not None:
                qdot_sol[self._joints_binding.dofadr] = joint_vel
            else:
                break

            # The velocity is pased to mujoco to be integrated.
            mjlib.mj_integratePos(
                self._physics.model.ptr,
                self._physics.data.qpos,
                qdot_sol,
                _INTEGRATION_TIMESTEP_SEC,
            )
            self._update_physics_data()

            # Compute error.
            linear_err = float(np.linalg.norm(site_xpos - target_position))

            # Stop if close enough to target.
            if early_stop and linear_err <= linear_tol:
                break

        qpos = np.array(self._joints_binding.qpos)
        return _Solution(qpos=qpos, linear_err=linear_err)

    def _compute_joint_velocities(
        self,
        site_id: int,
        target_position: np.ndarray,
        site_xpos: np.ndarray,
        regularization_weight: float,
    ) -> Optional[np.ndarray]:
        """Solves for joint velocities using damped least squares."""
        # Compute Jacobian.
        jacobian = np.empty((3, self._physics.model.nv))
        mjlib.mj_jacSite(
            self._physics.model.ptr,
            self._physics.data.ptr,
            jacobian,
            None,  # We don't care about rotation right now.
            site_id,
        )

        # Only grab the Jacobian values for the controllable joints.
        indexer = self._physics.named.model.dof_jntid.axes.row
        joint_names = [j.full_identifier for j in self._controllable_joints]
        dof_indices = indexer.convert_key_item(joint_names)
        jacobian_joints = jacobian[:, dof_indices]

        # Damped pseudoinverse.
        cartesian_delta = target_position - site_xpos
        if regularization_weight > 0.0:
            hess_approx = jacobian_joints.T @ jacobian_joints
            joint_delta = jacobian_joints.T @ cartesian_delta
            hess_approx += np.eye(hess_approx.shape[0]) * regularization_weight
            joint_vel = np.linalg.solve(hess_approx, joint_delta)
        else:
            joint_vel = np.linalg.lstsq(jacobian_joints, cartesian_delta, rcond=None)[0]

        return joint_vel

    def _update_physics_data(self) -> None:
        """Updates the physics data following the integration of velocities."""
        # Clip joint positions.
        qpos = self._joints_binding.qpos
        min_range = self._joints_binding.range[:, 0]
        max_range = self._joints_binding.range[:, 1]
        qpos = np.clip(qpos, min_range, max_range)
        self._joints_binding.qpos[:] = qpos

        # Forward kinematics to update the pose of the tracked site.
        mjlib.mj_normalizeQuat(self._physics.model.ptr, self._physics.data.qpos)
        mjlib.mj_kinematics(self._physics.model.ptr, self._physics.data.ptr)
        mjlib.mj_comPos(self._physics.model.ptr, self._physics.data.ptr)
