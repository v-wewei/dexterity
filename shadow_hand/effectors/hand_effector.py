import numpy as np
from dm_control import mjcf
from dm_env import specs

from shadow_hand import effector, hand
from shadow_hand.effectors import mujoco_actuation


class HandEffector(effector.Effector):
    """An effector interface for a dexterous hand."""

    def __init__(
        self,
        hand: hand.Hand,
        hand_name: str,
    ) -> None:
        self._hand = hand
        self._effector_prefix = f"{hand_name}_joint"
        self._mujoco_effector = mujoco_actuation.MujocoEffector(
            actuators=self._hand.actuators,
            prefix=self._effector_prefix,
        )

    @property
    def prefix(self) -> str:
        return self._mujoco_effector.prefix

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return self._mujoco_effector.action_spec(physics)

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        self._mujoco_effector.set_control(physics=physics, command=command)

    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        pass

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        pass
