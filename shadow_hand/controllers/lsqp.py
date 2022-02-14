import dataclasses

import numpy as np
from dm_control.mujoco.wrapper import mjbindings

from shadow_hand import hints
from shadow_hand.controllers.mapper import CartesianVelocitytoJointVelocityMapper
from shadow_hand.controllers.parameters import Parameters

mjlib = mjbindings.mjlib
enums = mjbindings.enums


@dataclasses.dataclass
class LSQPParameters(Parameters):
    ...


@dataclasses.dataclass
class LSQPMapper(CartesianVelocitytoJointVelocityMapper):

    params: LSQPParameters

    def compute_joint_velocities(
        self,
        data: hints.MjData,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        ...
