from typing import Optional

import dm_env

from shadow_hand.wrappers import base


class StepLimitWrapper(base.EnvironmentWrapper):
    def __init__(
        self, environment: dm_env.Environment, step_limit: Optional[int] = None
    ) -> None:
        super().__init__(environment)
        self._step_limit = step_limit
        self._elapsed_steps = 0

    def reset(self) -> dm_env.TimeStep:
        self._elapsed_steps = 0
        return self._environment.reset()

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        self._elapsed_steps += 1
        if self._step_limit is not None and self._elapsed_steps >= self._step_limit:
            return dm_env.truncation(
                timestep.reward, timestep.observation, timestep.discount
            )
        return timestep
