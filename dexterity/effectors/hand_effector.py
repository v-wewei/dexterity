import numpy as np
from dm_control import mjcf
from dm_env import specs

from dexterity import effector
from dexterity.effectors import mujoco_actuation
from dexterity.models.hands import fingered_hand


class HandEffector(effector.Effector):
    def __init__(
        self,
        hand: fingered_hand.FingeredHand,
        hand_name: str,
    ) -> None:
        self._hand = hand
        self._effector_prefix = f"{hand_name}_joint_tendon"

        self._mujoco_effector = mujoco_actuation.MujocoEffector(
            actuators=self._hand.actuators,
            prefix=self._effector_prefix,
        )

    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        pass

    def initialize_episode(self, physics, random_state) -> None:
        pass

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return self._mujoco_effector.action_spec(physics)

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        self._mujoco_effector.set_control(physics, command)

    @property
    def prefix(self) -> str:
        return self._mujoco_effector.prefix
