"""Common type hints for dexterity."""

from typing import Sequence, Tuple, Union

import mujoco
import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper.core import MjData
from dm_control.mujoco.wrapper.core import MjModel
from typing_extensions import TypeAlias

# General.
FloatArray = Union[Sequence[float], np.ndarray]
RgbaColor = Tuple[float, float, float, float]

# MuJoCo related.
MjcfElement: TypeAlias = mjcf.element._ElementImpl
MjcfAttachmentFrame: TypeAlias = mjcf.element._AttachmentFrame
MujocoModel: TypeAlias = MjModel
MujocoData: TypeAlias = MjData
MujocoObjectType: TypeAlias = mujoco.mjtObj
