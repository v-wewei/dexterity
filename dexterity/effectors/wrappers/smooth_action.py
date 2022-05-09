from typing import Optional

import numpy as np
from dm_control import mjcf

from dexterity import effector
from dexterity.effectors.wrappers import base


class ExponentialSmoother:
    """An exponential moving average."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("`alpha` must be between 0.0 and 1.0")

        self._alpha = alpha
        self._counter = 0
        self._value: Optional[np.ndarray] = None

    def update(self, value: np.ndarray) -> None:
        if self._counter == 0:
            self._value = value
        else:
            assert self._value is not None
            self._value = self._alpha * self._value + (1 - self._alpha) * value
        self._counter += 1

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def smoothed_value(self) -> np.ndarray:
        if self._value is None:
            raise ValueError("No value has been seen yet.")
        return self._value


class SmoothAction(base.Wrapper):
    """Smoothes the action using an exponential moving average."""

    def __init__(self, effector: effector.Effector, alpha: float) -> None:
        super().__init__(effector)

        self._alpha = alpha
        self._ema = ExponentialSmoother(alpha)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        # Reset the exponential moving average.
        self._ema = ExponentialSmoother(alpha=self._alpha)

        super().initialize_episode(physics, random_state)

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        # Update the exponential moving average.
        self._ema.update(command)

        # Get the smoothed action and forward it to the effector.
        super().set_control(physics, self._ema.smoothed_value)

    @property
    def ema(self) -> ExponentialSmoother:
        return self._ema
