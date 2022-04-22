import dm_env
import numpy as np
from dm_env import specs

from dexterity.manipulation.wrappers import base


# Adapted from https://github.com/deepmind/dm_control/blob/main/dm_control/suite/wrappers/action_noise.py
class ActionNoise(base.Wrapper):
    """A wrapper that adds Gaussian noise to the actions."""

    def __init__(self, environment: dm_env.Environment, scale: float):
        super().__init__(environment)

        action_spec = environment.action_spec()
        if not isinstance(action_spec, specs.BoundedArray):
            raise ValueError("Only `BoundedArray` action specs are supported.")

        self._minimum = action_spec.minimum
        self._maximum = action_spec.maximum
        self._noise_stddev = scale * (action_spec.maximum - action_spec.minimum)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        noisy_action = action + self.random_state.normal(scale=self._noise_stddev)
        np.clip(noisy_action, self._minimum, self._maximum, out=noisy_action)
        return self.environment.step(noisy_action)
