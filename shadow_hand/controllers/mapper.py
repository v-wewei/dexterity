import abc
import dataclasses
from typing import Sequence

import numpy as np
from dm_control.mujoco.wrapper import mjbindings

from shadow_hand import hints
from shadow_hand.utils import mujoco_utils

mjlib = mjbindings.mjlib
enums = mjbindings.enums


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
        data: hints.MjData,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        """Maps the Cartesian target velocity to joint velocities."""


@dataclasses.dataclass
class Parameters:
    """Container for `CartesianVelocitytoJointVelocityMapper` parameters."""

    model: hints.MjModel
    """MuJoCo `MjModel` instance."""

    joint_ids: Sequence[int]
    """MuJoCo joint IDs being controlled. Only 1 DoF joints are currently supported."""

    object_type: mjbindings.enums.mjtObj
    """MuJoCo type of the object being controlled. Only bodies, geoms and sites are
    currently supported."""

    object_name: str
    """MuJoCo name of the object being controlled."""

    def validate_parameters(self) -> None:
        """Validates the parameters."""

        if len(self.joint_ids) == 0:
            raise ValueError("At least one joint must be controlled.")

        for joint_id in self.joint_ids:
            if joint_id < 0 or joint_id >= self.model.njnt:
                raise ValueError(
                    f"Provided joint_id {joint_id} is invalid for the provided model, "
                    f"which has {self.model.njnt} joints."
                )

        ndof = len(mujoco_utils.joint_ids_to_dof_ids(self.model, self.joint_ids))
        if ndof != len(self.joint_ids):
            raise ValueError(
                f"`joint_ids` must only contain 1 DoF joints. "
                f"Number of joints: {len(self.joint_ids)}, number of dof: {ndof}."
            )

        if self.object_type not in [
            enums.mjtObj.mjOBJ_BODY,
            enums.mjtObj.mjOBJ_GEOM,
            enums.mjtObj.mjOBJ_SITE,
        ]:
            raise ValueError(
                f"Objects of type {self.object_type} are not supported. Only bodies, "
                "geoms and sites are supported."
            )

        if self.model.name2id(self.object_name, self.object_type) < 0:
            raise ValueError(
                f"Could not find MuJoCo object with name {self.object_name} and type "
                f"{self.object_type} in the provided model."
            )
