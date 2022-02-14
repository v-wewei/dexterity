import dataclasses

import numpy as np
from dm_control.mujoco.wrapper import mjbindings

from shadow_hand import hints
from shadow_hand.controllers import mapper
from shadow_hand.utils import mujoco_utils

mjlib = mjbindings.mjlib
enums = mjbindings.enums

# Alias.
PseudoInverseParameters = mapper.Parameters


@dataclasses.dataclass
class PseudoInverseMapper(mapper.CartesianVelocitytoJointVelocityMapper):
    """A `CartesianVelocitytoJointVelocityMapper` that uses the pseudoinverse of
    the Jacobian to map from Cartesian 6D velocity to joint velocities.

    Specifically, it maps the desired Cartesian velocity V to desired joint velocities
    v using the relationship:

    `v = [J]^+ @ V`

    where + denotes the pseudoinverse, @ denotes the matrix product, and J is the
    end-effector Jacobian.

    Note: This mapper shouldn't really be used since it performs very poorly near
    singularities. It is only implemented for pedagogical purposes.
    """

    params: PseudoInverseParameters

    def __post_init__(self) -> None:
        self.params.validate_parameters()

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
        # Could use: `np.linalg.pinv(jacobian_joints) @ target_velocity``, but better
        # not to explicity compute the pseudoinverse.
        return np.linalg.lstsq(jacobian_joints, target_velocity, rcond=None)[0]
