import dataclasses

import numpy as np
from dm_control.mujoco.wrapper import mjbindings

from shadow_hand import hints
from shadow_hand.controllers import mapper
from shadow_hand.utils import mujoco_utils

mjlib = mjbindings.mjlib
enums = mjbindings.enums


@dataclasses.dataclass
class DampedLeastSquaresParameters(mapper.Parameters):

    regularization_weight: float
    """Damping factor used to regularize the pseudoinverse."""

    def validate_parameters(self) -> None:
        if self.regularization_weight < 0:
            raise ValueError(
                "`regularization_weight` must be non-negative, but was "
                f"{self.regularization_weight}."
            )

        super().validate_parameters()


@dataclasses.dataclass
class DampedLeastSquaresMapper(mapper.CartesianVelocitytoJointVelocityMapper):
    """A `CartesianVelocitytoJointVelocityMapper` that uses damped least-squares to
    solve for the joint velocities.

    Specifically, it maps the desired Cartesian velocity V to desired joint velocities
    v using the relationship:

    `v = [J^T @ J + lambda * I] @ V`

    where @ denotes the matrix product, J is the end-effector Jacobian and lambda is
    a damping factor.
    """

    params: DampedLeastSquaresParameters

    def __post_init__(self) -> None:
        self.params.validate_parameters()

    def compute_joint_velocities(
        self,
        data: hints.MjData,
        target_velocity: np.ndarray,
        # nullspace_bias: np.ndarray,
    ) -> np.ndarray:
        # Compute the Jacobian.
        jacobian = mujoco_utils.compute_object_6d_jacobian(
            self.params.model,
            data,
            self.params.object_type,
            self.params.model.name2id(self.params.object_name, self.params.object_type),
        )

        # Only grab the Jacobian values for the controllable joints.
        # We're also ignoring the angular values for now.
        jacobian_joints = jacobian[:3, self.params.joint_ids]

        hess_approx = jacobian_joints.T @ jacobian_joints
        joint_delta = jacobian_joints.T @ target_velocity
        hess_approx += np.eye(hess_approx.shape[0]) * self.params.regularization_weight
        return np.linalg.solve(hess_approx, joint_delta)

        # jacobian_pinv = np.linalg.pinv(jacobian_joints)
        # solution = jacobian_pinv @ target_velocity
        # if nullspace_bias is not None:
        #     solution += (
        #         np.eye(jacobian_pinv.shape[0]) - jacobian_pinv @ jacobian_joints
        #     ) @ nullspace_bias
        # return solution
