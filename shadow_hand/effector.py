import abc

import numpy as np
from dm_control import mjcf
from dm_env import specs


class Effector(abc.ABC):
    """Abstract effector interface.

    An effector provides an interface for an agent to interact with the environment.
    The effector is defined by its action spec, a control method, and a prefix that
    marks the control components of the effector in the wider task action spec.
    """

    @property
    @abc.abstractmethod
    def prefix(self) -> str:
        ...

    @abc.abstractmethod
    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        ...

    @abc.abstractmethod
    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        ...

    @abc.abstractmethod
    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        ...

    @abc.abstractmethod
    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        ...
