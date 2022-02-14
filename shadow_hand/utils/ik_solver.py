"""Inverse kinematics solver."""

import copy
import dataclasses
from typing import Optional, Sequence

import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings
from dm_robotics.geometry import geometry, mujoco_physics

from shadow_hand import controllers, hints
from shadow_hand.utils import geometry_utils, mujoco_utils

mjlib = mjbindings.mjlib

# Gain for the linear and angular twist computation, these values should always
# be between 0 and 1. 0 corresponds to not moving and 1 corresponds to moving to the
# target in a single integration timestep.
_LINEAR_VELOCITY_GAIN = 0.95
_ANGULAR_VELOCITY_GAIN = 0.95

# Integration timestep used to convert from joint velocities to joint positions.
_INTEGRATION_TIMESTEP_SEC = 1.0

# Damping factor.
_REGULARIZATION_WEIGHT = 1e-3

# If the norm of the error divided by the magnitude of the joint position update is
# greater than this value, then the solve is ended prematurely. This helps us avoid
# getting stuck in local minima.
_PROGRESS_THRESHOLD = 20.0


@dataclasses.dataclass
class _Solution:
    """Return value of an IK solution."""

    qpos: np.ndarray
    linear_err: float


class IKSolver:
    """Inverse kinematics solver.

    Computes a joint configuration that brings an element to a certain pose.
    """

    def __init__(
        self,
        model: mjcf.RootElement,
        controllable_joints: Sequence[hints.MjcfElement],
        element: hints.MjcfElement,
    ) -> None:
        """Constructor.

        Args:
            model: The MJCF model root.
            controllable_joints: The joints that can be controlled to achieve the
                desired pose. Only 1 DoF joints are supported.
            element: The MJCF element that is being controlled. Only bodys, geoms and
                sites are supported.
        """
        self._physics = mjcf.Physics.from_mjcf_model(model)
        self._geometry_physics = mujoco_physics.wrap(self._physics)
        self._joints_binding = self._physics.bind(controllable_joints)
        self._controllable_joints = controllable_joints
        self._num_joints = len(self._controllable_joints)
        self._element = element
        self._create_mapper()

    def _create_mapper(self) -> None:
        """Creates the Cartesian velocity to joint velocities mapper."""
        self._joint_ids = self._joints_binding.jntid
        self._object_type = mujoco_utils.get_element_type(self._element)
        self._object_name = self._element.full_identifier

        params = controllers.dls.DampedLeastSquaresParameters(
            model=self._physics.model,
            joint_ids=self._joints_binding.jntid,
            object_type=mujoco_utils.get_element_type(self._element),
            object_name=self._element.full_identifier,
            regularization_weight=_REGULARIZATION_WEIGHT,
        )
        self._mapper = controllers.dls.DampedLeastSquaresMapper(params)

        self._joints_argsort = np.argsort(self._joints_binding.jntid)

        # Default nullspace reference is the mid-range of the joints.
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
    ) -> Optional[np.ndarray]:
        """Attempts to solve the inverse kinematics.

        Returns None if no solution is found. If multiple solutions are found, returns
        the one where the joints are closer to the `nullspace_reference`. If no such
        `nullspace_reference` is provided, defaults to the center of the joint ranges.
        """
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

        # Each iteration of this loop attempts to solve the IK problem.
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
            )

            # Check if the attempt was successful. The solution is saved if the joints
            # are closer to the nullspace reference.
            if sol.linear_err <= linear_tol:
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
            print("IK did not find a solution.")

        return sol_qpos

    def _solve_ik(
        self,
        target_position: np.ndarray,
        linear_tol: float,
        max_steps: int,
        early_stop: bool,
    ) -> _Solution:
        """Solves for a joint configuration that brings element pose to target pose."""
        cur_frame = geometry.PoseStamped(pose=None, frame=self._element)
        cur_pose = cur_frame.get_world_pose(self._geometry_physics)
        previous_pose = copy.copy(cur_pose)
        linear_err: float = np.inf

        # Each iteration of this loop attempts to reduce the error between the site's
        # position and the target position.
        for _ in range(max_steps):
            twist = _compute_twist(
                cur_pose,
                geometry.Pose(position=target_position, quaternion=None),
                _LINEAR_VELOCITY_GAIN,
                _ANGULAR_VELOCITY_GAIN,
                _INTEGRATION_TIMESTEP_SEC,
            )

            # Computes the joint velocities that achieve the desired twist.
            # The joint velocity vector passed to mujoco's integration needs to have a
            # value for all the joints in the model. The velocity for all the joints
            # that are not controlled is set to 0.
            qdot_sol = np.zeros(self._physics.model.nv)
            joint_vel = self._compute_joint_velocities(twist.linear)

            # If we are unable to compute joint velocities, we stop the iteration as the
            # solver is stuck and cannot make any more progress.
            if joint_vel is not None:
                qdot_sol[self._joints_binding.dofadr] = joint_vel
            else:
                break

            mjbindings.mjlib.mj_integratePos(
                self._physics.model.ptr,
                self._physics.data.qpos,
                qdot_sol,
                _INTEGRATION_TIMESTEP_SEC,
            )
            self._update_physics_data()

            # Get the distance and the angle between the current pose and the target
            # pose.
            cur_pose = cur_frame.get_world_pose(self._geometry_physics)
            linear_err = float(np.linalg.norm(target_position - cur_pose.position))

            # Stop if the pose is close enough to the target pose.
            if early_stop and (linear_err <= linear_tol):
                break

            # We measure the progress made during this step. If the error is not reduced
            # fast enough the solve is stopped to save computation time.
            linear_change = np.linalg.norm(cur_pose.position - previous_pose.position)
            if linear_err / (linear_change + 1e-10) > _PROGRESS_THRESHOLD:
                break

            previous_pose = copy.copy(cur_pose)

        qpos = np.array(self._joints_binding.qpos)
        return _Solution(qpos=qpos, linear_err=linear_err)

    def _compute_joint_velocities(
        self, cartesian_6d_target: np.ndarray
    ) -> Optional[np.ndarray]:
        """Maps a Cartesian 6D target velocity to joint velocities."""
        try:
            joint_velocities = np.empty(self._num_joints)
            jvel = self._mapper.compute_joint_velocities(
                self._physics.data,
                cartesian_6d_target,
            )
            joint_velocities[self._joints_argsort] = jvel
            return joint_velocities
        except Exception:
            return None

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


def _compute_twist(
    init_pose: geometry.Pose,
    final_pose: geometry.Pose,
    linear_velocity_gain: float,
    angular_velocity_gain: float,
    control_timestep_seconds: float,
) -> geometry.Twist:
    """Returns the twist to apply to the element to reach final_pose from init_pose."""
    position_error = final_pose.position - init_pose.position
    orientation_error = geometry_utils.get_orientation_error(
        to_quat=final_pose.quaternion,
        from_quat=init_pose.quaternion,
    )
    linear = linear_velocity_gain * position_error / control_timestep_seconds
    angular = angular_velocity_gain * orientation_error / control_timestep_seconds
    return geometry.Twist(np.concatenate((linear, angular)))
