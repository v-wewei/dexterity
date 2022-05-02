import numpy as np
from dm_control import mjcf
from dm_env import specs

from dexterity import effector


class Wrapper(effector.Effector):
    """Base class for effector wrappers."""

    def __init__(self, effector: effector.Effector) -> None:
        self._delegate_effector = effector

    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        self._delegate_effector.after_compile(mjcf_model)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._delegate_effector.initialize_episode(physics, random_state)

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return self._delegate_effector.action_spec(physics)

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        self._delegate_effector.set_control(physics, command)

    @property
    def prefix(self) -> str:
        return self._delegate_effector.prefix
