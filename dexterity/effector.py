import abc

import numpy as np
from dm_control import mjcf
from dm_env import specs


class Effector(abc.ABC):
    """Abstract base class for an effector."""

    @abc.abstractmethod
    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        ...

    @abc.abstractmethod
    def initialize_episode(
        self,
        physics: mjcf.Physics,
        random_state: np.random.RandomState,
    ) -> None:
        ...

    @abc.abstractmethod
    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        ...

    @abc.abstractmethod
    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        ...

    @property
    @abc.abstractmethod
    def prefix(self) -> str:
        ...
