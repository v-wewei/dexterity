from typing import Optional, Sequence

import dm_env
import numpy as np
import tree

from shadow_hand.wrappers import base


class ConcatObservationWrapper(base.EnvironmentWrapper):
    def __init__(
        self,
        environment: dm_env.Environment,
        name_filter: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(environment)
        observation_spec = environment.observation_spec()
        if name_filter is None:
            name_filter = list(observation_spec.keys())
        self._obs_names = [x for x in name_filter if x in observation_spec.keys()]

        dummy_obs = _zeros_like(observation_spec)
        dummy_obs = self._convert_observation(dummy_obs)
        self._observation_spec = dm_env.specs.BoundedArray(
            shape=dummy_obs.shape,
            dtype=dummy_obs.dtype,
            minimum=-np.inf,
            maximum=np.inf,
            name="state",
        )

    def _convert_observation(self, observation):
        obs = {k: observation[k] for k in self._obs_names}
        return _concat(obs)

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return timestep._replace(
            observation=self._convert_observation(timestep.observation)
        )

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        return timestep._replace(
            observation=self._convert_observation(timestep.observation)
        )

    def observation_spec(self):
        return self._observation_spec


def _zeros_like(nest, dtype=None):
    return tree.map_structure(lambda x: np.zeros(x.shape, dtype or x.dtype), nest)


def _concat(values) -> np.ndarray:
    leaves = list(map(np.atleast_1d, tree.flatten(values)))
    return np.concatenate(leaves)
