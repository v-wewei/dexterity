from typing import Callable

import dm_env
import tree

from shadow_hand.wrappers import base


class TransformObservationWrapper(base.EnvironmentWrapper):
    def __init__(
        self,
        environment: dm_env.Environment,
        transform_func: Callable,
    ) -> None:
        super().__init__(environment)
        self._transform_func = transform_func

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return timestep._replace(
            observation=tree.map_structure(self._transform_func, timestep.observation)
        )

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        return timestep._replace(
            observation=tree.map_structure(self._transform_func, timestep.observation)
        )


class TransformRewardWrapper(base.EnvironmentWrapper):
    def __init__(
        self,
        environment: dm_env.Environment,
        transform_func: Callable,
    ) -> None:
        super().__init__(environment)
        self._transform_func = transform_func

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return timestep._replace(reward=self._transform_func(timestep.reward))
