import dataclasses
from typing import Sequence

from dm_control.mujoco.wrapper import mjbindings

from shadow_hand import hints
from shadow_hand.utils import mujoco_utils

mjlib = mjbindings.mjlib
enums = mjbindings.enums


@dataclasses.dataclass
class Parameters:
    """Container for `CartesianVelocitytoJointVelocityMapper` parameters."""

    model: hints.MjModel
    """MuJoCo `MjModel` instance."""

    joint_ids: Sequence[int]
    """MuJoCo joint IDs being controlled."""

    object_type: mjbindings.enums.mjtObj
    """MuJoCo type of the object being controlled."""

    object_name: str
    """MuJoCo name of the object being controlled."""

    integration_timestep: float
    """Amount of time the joint velocities will be executed for, aka `dt`. This timestep
    will be used to ensure safety contraints are not violated. Higher values are more
    conservative."""

    def __post_init__(self) -> None:
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

        if self.integration_timestep <= 0:
            raise ValueError(
                f"Integration timestep {self.integration_timestep} must be a positive "
                "duration."
            )
