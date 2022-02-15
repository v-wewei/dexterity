import copy
import dataclasses
from typing import Dict, List, Mapping, Optional

import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings
from dm_robotics.geometry import geometry, mujoco_physics

from shadow_hand import controllers
from shadow_hand.models.hands import shadow_hand_e_constants as consts
from shadow_hand.utils import geometry_utils, mujoco_utils

mjlib = mjbindings.mjlib

_FINGERS: List[consts.Components] = [
    consts.Components.TH,
    consts.Components.FF,
    consts.Components.MF,
    consts.Components.RF,
    consts.Components.LF,
]

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
    """Inverse kinematics solver for a dexterous hand."""

    def __init__(self, model: mjcf.RootElement, prefix: str = "") -> None:
        """Constructor.

        Args:
            model: The MJCF model root.
            prefix: The prefix assigned to the hand model in case it is attached to
                another entity.
        """
        self._physics = mjcf.Physics.from_mjcf_model(model)
        self._geometry_physics = mujoco_physics.wrap(self._physics)

        # Finger information.
        self._controllable_joints = {}
        self._elements = {}
        for finger in _FINGERS:
            fingertip_name = finger.name.lower() + "tip"
            fingertip_site_name = f"{prefix}/{fingertip_name}_site"
            fingertip_site_elem = model.find("site", fingertip_site_name)
            assert fingertip_site_elem is not None
            self._elements[finger] = fingertip_site_elem

            joint_names = [j.name for j in consts.JOINT_GROUP[finger]]
            controllable_joints = []
            for joint_name in joint_names:
                joint_elem = model.find("joint", f"{prefix}/{joint_name}")
                controllable_joints.append(joint_elem)
            assert len(controllable_joints) == len(joint_names)
            self._controllable_joints[finger] = controllable_joints

        self._joint_bindings = {}
        self._num_joints = {}
        for finger, controllable_joints in self._controllable_joints.items():
            self._joint_bindings[finger] = self._physics.bind(controllable_joints)
            self._num_joints[finger] = len(controllable_joints)

        # Wrist information.
        wrist_joint_names = [j.name for j in consts.JOINT_GROUP[consts.Components.WR]]
        self._wrist_controllable_joints = []
        for joint_name in wrist_joint_names:
            joint_elem = model.find("joint", f"{prefix}/{joint_name}")
            self._wrist_controllable_joints.append(joint_elem)
        assert len(self._wrist_controllable_joints) == len(wrist_joint_names)
        self._wrist_num_joints = len(wrist_joint_names)
        self._wirst_joint_bindings = self._physics.bind(self._wrist_controllable_joints)
        self._wrist_nullspace_joint_position_reference = 0.5 * np.sum(
            self._wirst_joint_bindings.range, axis=1
        )

        self._create_mappers()

    def _create_mappers(self) -> None:
        self._mappers = {}
        self._joints_argsort = {}
        self._nullspace_joint_position_reference = {}
        for finger in _FINGERS:
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
    ) -> Dict[consts.Components, Optional[np.ndarray]]:
        """ "Attempts to solve the inverse kinematics."""
        # Set the initial wrist and finger configurations to zero.
        initial_wrist_configuration = np.zeros(self._wrist_num_joints)
        inital_joint_configuration = {}
        for finger, num_joints in self._num_joints.items():
            inital_joint_configuration[finger] = np.zeros(num_joints)

        # Set initial solutions to None.
        solutions: Dict[consts.Components, Optional[np.ndarray]] = {}
        for finger in target_positions.keys():
            solutions[finger] = None
        solutions[consts.Components.WR] = None

        nullspace_jnt_qpos_min_err: float = np.inf

        # Each iteration of this loop attempts to solve the IK problem.
        for attempt in range(num_attempts):
            # Randomize the initial joint configuration so that the IK can find
            # different solutions.
            if attempt == 0:
                wrist_configuration = initial_wrist_configuration
                for finger, joint_binding in self._joint_bindings.items():
                    joint_binding.qpos[:] = inital_joint_configuration[finger]
            else:
                wrist_configuration = np.random.uniform(
                    self._wirst_joint_bindings.range[:, 0],
                    self._wirst_joint_bindings.range[:, 1],
                )
                for finger, joint_binding in self._joint_bindings.items():
                    joint_binding.qpos[:] = np.random.uniform(
                        joint_binding.range[:, 0], joint_binding.range[:, 1]
                    )
            self._wirst_joint_bindings.qpos[:] = wrist_configuration

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
                nullspace_jnt_qpos_err += float(
                    np.linalg.norm(
                        wrist_configuration
                        - self._wrist_nullspace_joint_position_reference
                    )
                )

                if nullspace_jnt_qpos_err < nullspace_jnt_qpos_min_err:
                    nullspace_jnt_qpos_min_err = nullspace_jnt_qpos_err
                    for finger, solution in finger_solutions.items():
                        solutions[finger] = solution.qpos
                    solutions[consts.Components.WR] = wrist_configuration

            if all_success and stop_on_first_successful_attempt:
                break

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
            joint_vel = self._compute_joint_velocities(finger, twist.linear)

            # If we are unable to compute joint velocities, we stop the iteration as the
            # solver is stuck and cannot make any more progress.
            if joint_vel is not None:
                qdot_sol[self._joint_bindings[finger].dofadr] = joint_vel
            else:
                break

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
    ) -> Optional[np.ndarray]:
        """Maps a Cartesian 6D target velocity to joint velocities."""
        try:
            joint_velocities = np.empty(self._num_joints[finger])
            jvel = self._mappers[finger].compute_joint_velocities(
                self._physics.data,
                cartesian_6d_target,
            )
            joint_velocities[self._joints_argsort[finger]] = jvel
            return joint_velocities
        except Exception:
            return None

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
