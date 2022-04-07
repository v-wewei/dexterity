from typing import Any

import dm_env


class EnvironmentWrapper(dm_env.Environment):
    def __init__(self, environment: dm_env.Environment) -> None:
        self._environment = environment

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(f"Attempted to access private attribute {name}")
        return getattr(self._environment, name)

    @property
    def environment(self) -> dm_env.Environment:
        return self._environment

    def step(self, action) -> dm_env.TimeStep:
        return self._environment.step(action)

    def reset(self) -> dm_env.TimeStep:
        return self._environment.reset()

    def action_spec(self):
        return self._environment.action_spec()

    def discount_spec(self):
        return self._environment.discount_spec()

    def observation_spec(self):
        return self._environment.observation_spec()

    def reward_spec(self):
        return self._environment.reward_spec()

    def close(self):
        return self._environment.close()
