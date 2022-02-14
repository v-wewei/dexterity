import abc

import numpy as np

from shadow_hand.hints import MjData


class CartesianVelocitytoJointVelocityMapper(abc.ABC):
    """Abstract base class for a Cartesian velocity to joint velocity mapper.

    Subclasses should implement the `compute_joint_velocities` method, which maps a
    Cartesian 6D velocity about the global frame to joint velocities. The target
    Cartesian velocity is specified as a 6D vector, with the 3D linear velocity term
    followed by the 3D angular velocity term.

    At every call to `compute_joint_velocities`, the mapper will attempt to solve for
    the joint velocities that realize the target Cartesian velocity on the frame
    attached to the MuJoCo object defined by `object_type` and `object_name`. The target
    Cartesian velocity must be expressed about the MuJoCo object's origin, in world
    orientation. An error is returned if the computed velocities do not exist, or if the
    mapper fails to compute the velocities.

    This class expects an updated MjData object at every call to
    `compute_joint_velocities`, so users should ensure that the mjData object has
    up-to-date `qpos` and `qvel` fields.
    """

    @abc.abstractmethod
    def compute_joint_velocities(
        self,
        data: MjData,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        """Maps the Cartesian target velocity to joint velocities."""
