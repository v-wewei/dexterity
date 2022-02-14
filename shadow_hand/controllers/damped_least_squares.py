import dataclasses

import numpy as np
from dm_control.mujoco.wrapper import mjbindings

from shadow_hand import hints
from shadow_hand.controllers.mapper import CartesianVelocitytoJointVelocityMapper
from shadow_hand.controllers.parameters import Parameters
from shadow_hand.utils import mujoco_utils

mjlib = mjbindings.mjlib
enums = mjbindings.enums


@dataclasses.dataclass
class DampedLeastSquaresParameters(Parameters):

    regularization_weight: float
    """Damping factor used to regularize the pseudoinverse."""


@dataclasses.dataclass
class DampedLeastSquaresMapper(CartesianVelocitytoJointVelocityMapper):
    """A `CartesianVelocitytoJointVelocityMapper` that uses damped least-squares or
    Levenberg-Marquardt to solve for the joint velocities.

    Specifically, it maps the desired Cartesian velocity V to desired joint velocities
    v using the relationship:

    `v = [J^T @ J + lambda * I] @ V`

    where @ denotes the matrix product, J is the end-effector Jacobian and lambda is
    a damping factor.
    """

    params: DampedLeastSquaresParameters

    def compute_joint_velocities(
        self,
        data: hints.MjData,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        # Compute the Jacobian.
        jacobian = mujoco_utils.compute_object_6d_jacobian(
            self.params.model,
            data,
            self.params.object_type,
            self.params.model.name2id(self.params.object_name, self.params.object_type),
        )

        # Only grab the Jacobian values for the controllable joints.
        jacobian_joints = jacobian[:, self.params.joint_ids]

        # TODO(kevin): Add rotation component.
        jacobian_joints = jacobian_joints[:3]

        # Solve!
        hess_approx = jacobian_joints.T @ jacobian_joints
        joint_delta = jacobian_joints.T @ target_velocity
        hess_approx += np.eye(hess_approx.shape[0]) * self.params.regularization_weight
        solution = np.linalg.solve(hess_approx, joint_delta)
        return solution
