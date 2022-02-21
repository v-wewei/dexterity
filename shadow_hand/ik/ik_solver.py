import copy
import dataclasses
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings
from dm_robotics.geometry import geometry, mujoco_physics

from shadow_hand import controllers
from shadow_hand.models.hands import shadow_hand_e_constants as consts
from shadow_hand.utils import geometry_utils, mujoco_utils

mjlib = mjbindings.mjlib

# Gain for the linear and angular twist computation, these values should always
# be between 0 and 1. 0 corresponds to not moving and 1 corresponds to moving to the
# target in a single integration timestep.
_LINEAR_VELOCITY_GAIN = 0.95
_ANGULAR_VELOCITY_GAIN = 0.95

_NULLSPACE_GAIN = 0.4

# Integration timestep used to convert from joint velocities to joint positions.
_INTEGRATION_TIMESTEP_SEC = 1.0

# Damping factor.
_REGULARIZATION_WEIGHT = 1e-5

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
    """Inverse kinematics solver for a dexterous hand."""

    def __init__(
        self,
        model: mjcf.RootElement,
        fingers: Sequence[consts.Components],
        prefix: str = "",
    ) -> None:
        """Constructor.

        Args:
            model: The MJCF model root.
            prefix: The prefix assigned to the hand model in case it is attached to
                another entity.
            nullspace_gain: Scales the nullspace velocity bias. If the gain is set to 0,
                there will be no nullspace optimization during the solve process.
        """
        self._fingers = fingers
        self._physics = mjcf.Physics.from_mjcf_model(model)
        self._geometry_physics = mujoco_physics.wrap(self._physics)

        # Wrist information.
        wrist_joint_names = [j.name for j in consts.JOINT_GROUP[consts.Components.WR]]

        # Finger information.
        self._controllable_joints = {}
        self._elements = {}
        for finger in fingers:
            fingertip_name = consts.FINGER_FINGERTIP_MAPPING[finger]
            fingertip_site_name = mujoco_utils.prefix_identifier(
                f"{fingertip_name}_site", prefix
            )
            fingertip_site_elem = model.find("site", fingertip_site_name)
            assert fingertip_site_elem is not None
            self._elements[finger] = fingertip_site_elem

            joint_names = [j.name for j in consts.JOINT_GROUP[finger]]
            joint_names += wrist_joint_names
            controllable_joints = []
            for joint_name in joint_names:
                joint_elem = model.find(
                    "joint", mujoco_utils.prefix_identifier(joint_name, prefix)
                )
                controllable_joints.append(joint_elem)
            assert len(controllable_joints) == len(joint_names)
            self._controllable_joints[finger] = controllable_joints

        self._joint_bindings = {}
        self._num_joints = {}
        for finger, controllable_joints in self._controllable_joints.items():
            self._joint_bindings[finger] = self._physics.bind(controllable_joints)
            self._num_joints[finger] = len(controllable_joints)

        self._create_mappers()

    def _create_mappers(self) -> None:
        self._mappers = {}
        self._joints_argsort = {}
        self._nullspace_joint_position_reference = {}
        for finger in self._fingers:
            params = controllers.dls.DampedLeastSquaresParameters(
                model=self._physics.model,
                joint_ids=self._joint_bindings[finger].jntid,
                object_type=mujoco_utils.get_element_type(self._elements[finger]),
                object_name=self._elements[finger].full_identifier,
                regularization_weight=_REGULARIZATION_WEIGHT,
            )
            self._mappers[finger] = controllers.dls.DampedLeastSquaresMapper(params)
            self._joints_argsort[finger] = np.argsort(
                self._joint_bindings[finger].jntid
            )
            self._nullspace_joint_position_reference[finger] = 0.5 * np.sum(
                self._joint_bindings[finger].range, axis=1
            )

    def solve(
        self,
        target_positions: Mapping[consts.Components, np.ndarray],
        linear_tol: float = 1e-3,
        max_steps: int = 100,
        early_stop: bool = False,
        num_attempts: int = 30,
        stop_on_first_successful_attempt: bool = False,
    ) -> Optional[Dict[consts.Components, np.ndarray]]:
        """Attempts to solve the inverse kinematics.

        Returns a mapping from finger to joint positions that achieve the desired
        target position. If one of the fingers does not find a solution, then returns
        None.

        Args:
            target_positions: A mapping from finger to desired controlled finger element
                target pose in the world frame.
            linear_tol: The linear tolerance, in meters, that determines if the IK
                solution is valid.
            max_steps: Maximum number of integration steps that can be used. The larger
                this value is, the more likely a solution will be found, at the
                expense of computation time.
            early_stop: If true, stops the attempt as soon as the configuration is
                within the linear tolerance. If false, it will always run `max_steps`
                iterations per attempt.
            num_attempts: The number of different attempts the solver should do. More
                attemps increase the probability of finding a solution that is closer
                to the nullspace reference. By default, this is set to all zeros.
            stop_on_first_successful_attempt: If true, the method will return the
                first solution where all finger solutions meet the tolerance criterion.
        """
        # Set the initial finger configuration to zero.
        inital_joint_configuration = {}
        for finger, num_joints in self._num_joints.items():
            inital_joint_configuration[finger] = np.zeros(num_joints)

        solutions: Dict[consts.Components, np.ndarray] = {}
        nullspace_jnt_qpos_min_err: float = np.inf

        # Each iteration of this loop attempts to solve the IK problem.
        for attempt in range(num_attempts):
            # Randomize the initial joint configuration so that the IK can find
            # different solutions.
            if attempt == 0:
                for finger, joint_binding in self._joint_bindings.items():
                    joint_binding.qpos[:] = inital_joint_configuration[finger]
            else:
                for finger, joint_binding in self._joint_bindings.items():
                    joint_binding.qpos[:] = np.random.uniform(
                        joint_binding.range[:, 0], joint_binding.range[:, 1]
                    )

            # Solve each finger separately.
            finger_solutions: Dict[consts.Components, _Solution] = {}
            all_success: bool = True
            for finger, target_position in target_positions.items():
                solution = self._solve_ik(
                    finger,
                    target_position,
                    linear_tol,
                    max_steps,
                    early_stop,
                )
                if solution.linear_err > linear_tol:
                    all_success = False
                finger_solutions[finger] = solution

            # Save the solution if closer to nullspace reference.
            if all_success:
                nullspace_jnt_qpos_err: float = 0.0
                for finger, solution in finger_solutions.items():
                    nullspace_jnt_qpos_err += float(
                        np.linalg.norm(
                            solution.qpos
                            - self._nullspace_joint_position_reference[finger]
                        )
                    )

                if nullspace_jnt_qpos_err < nullspace_jnt_qpos_min_err:
                    nullspace_jnt_qpos_min_err = nullspace_jnt_qpos_err
                    for finger, solution in finger_solutions.items():
                        solutions[finger] = solution.qpos

            if all_success and stop_on_first_successful_attempt:
                break

        if not solutions:
            print(f"{self.__class__.__name__} failed to find a solution.")
            # # Uncomment to see best solve attempt.
            # best_try = {}
            # for finger, solution in finger_solutions.items():
            #     best_try[finger] = solution.qpos
            # # best_try[consts.Components.WR] = wrist_configuration
            # return best_try
            return None

        return solutions

    def _solve_ik(
        self,
        finger: consts.Components,
        target_position: np.ndarray,
        linear_tol: float,
        max_steps: int,
        early_stop: bool,
    ) -> _Solution:
        """Solves for a joint configuration that brings element pose to target pose."""
        cur_frame = geometry.PoseStamped(pose=None, frame=self._elements[finger])
        cur_pose = cur_frame.get_world_pose(self._geometry_physics)
        previous_pose = copy.copy(cur_pose)
        linear_err: float = np.inf

        # Each iteration of this loop attempts to reduce the error between the site's
        # position and the target position.
        for i in range(max_steps):
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
            joint_vel = self._compute_joint_velocities(finger, twist.linear)

            qdot_sol[self._joint_bindings[finger].dofadr] = joint_vel

            mjbindings.mjlib.mj_integratePos(
                self._physics.model.ptr,
                self._physics.data.qpos,
                qdot_sol,
                _INTEGRATION_TIMESTEP_SEC,
            )
            self._update_physics_data(finger)

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

        qpos = np.array(self._joint_bindings[finger].qpos)
        return _Solution(qpos=qpos, linear_err=linear_err)

    def _compute_joint_velocities(
        self, finger: consts.Components, cartesian_6d_target: np.ndarray
    ) -> np.ndarray:
        """Maps a Cartesian 6D target velocity to joint velocities."""

        # nullspace_bias = (
        #     _NULLSPACE_GAIN
        #     * (
        #         self._nullspace_joint_position_reference[finger]
        #         - self._joint_bindings[finger].qpos
        #     )
        #     / _INTEGRATION_TIMESTEP_SEC
        # )

        return self._mappers[finger].compute_joint_velocities(
            data=self._physics.data,
            target_velocity=cartesian_6d_target,
            # nullspace_bias=nullspace_bias,
        )

    def _update_physics_data(self, finger: consts.Components) -> None:
        """Updates the physics data following the integration of velocities."""
        # Clip joint positions.
        qpos = self._joint_bindings[finger].qpos
        min_range = self._joint_bindings[finger].range[:, 0]
        max_range = self._joint_bindings[finger].range[:, 1]
        qpos = np.clip(qpos, min_range, max_range)
        self._joint_bindings[finger].qpos[:] = qpos

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
