import numpy as np
from dm_control import mjcf
from dm_env import specs

from shadow_hand import effector
from shadow_hand.effectors import hand_effector


class RelativeEffector(effector.Effector):
    def __init__(
        self,
        hand_effector: hand_effector.HandEffector,
    ) -> None:
        self._effector = hand_effector

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        delta_command = command - physics.bind(self._effector.hand.actuators).ctrl
        self._effector.set_control(physics, delta_command)

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return self._effector.action_spec(physics)

    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        return self._effector.after_compile(mjcf_model)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        return self._effector.initialize_episode(physics, random_state)

    @property
    def prefix(self) -> str:
        return self._effector.prefix
