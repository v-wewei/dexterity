"""Wraps a dexterity environment to be used as a Gym environment."""

import copy
from typing import Any, Dict, OrderedDict, Tuple, Union

import dm_env
import gym
import numpy as np
from dm_env import specs
from gym import spaces


class GymWrapper(gym.Env):
    """Environment wrapper for dexterity environments."""

    def __init__(self, environment: dm_env.Environment) -> None:
        self._environment = environment

        # Convert action and observation specs.
        self._action_space = _convert_to_space(environment.action_spec())
        self._observation_space = _convert_to_space(environment.observation_spec())

    def seed(self, seed):
        self._environment.random_state.seed(seed)
        return [seed]

    def reset(self):
        timestep = self._environment.reset()
        return timestep.observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        timestep = self._environment.step(action)

        obs = timestep.observation
        reward = timestep.reward or 0.0  # dm_env can return None for reward.
        done = timestep.last()
        info = {"discount": timestep.discount}

        return obs, reward, done, info

    def render(
        self, mode="rgb_array", height: int = 84, width: int = 84, camera_id: int = 0
    ) -> np.ndarray:
        if mode != "rgb_array":
            raise NotImplementedError("Only `rgb_array` mode is supported.")

        return self._environment.render(
            height=height,
            width=width,
            camera_id=camera_id,
        )

    def __getattr__(self, name: Any) -> Any:
        return getattr(self._environment, name)

    @property
    def environment(self) -> dm_env.Environment:
        """Returns the wrapped environment."""
        return self._environment

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space


def _convert_to_space(spec: specs.Array) -> spaces.Space:
    """Converts a dm_env spec to an OpenAI Gym space."""
    if isinstance(spec, OrderedDict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = _convert_to_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, specs.Array):
        low: Union[int, float]
        high: Union[int, float]
        if np.issubdtype(spec.dtype, np.integer):
            low = np.iinfo(spec.dtype).min
            high = np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low = float("-inf")
            high = float("inf")
        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    else:
        raise NotImplementedError(f"Unsupported spec type: {type(spec)}")
