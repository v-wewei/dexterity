import abc

import numpy as np
from dm_control import mjcf, specs


class Effector(abc.ABC):
    """Abstract effector interface, a controllable element of the environment."""

    @abc.abstractmethod
    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        ...

    @abc.abstractmethod
    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        ...
