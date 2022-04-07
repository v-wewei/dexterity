import dm_env
import numpy as np

from shadow_hand.wrappers import base


class ClipActionWrapper(base.EnvironmentWrapper):
    def __init__(self, environment: dm_env.Environment) -> None:
        assert isinstance(environment.action_spec(), dm_env.specs.BoundedArray)
        super().__init__(environment)

    def step(self, action) -> dm_env.TimeStep:
        action_spec = self.action_spec()
        action = np.clip(action, action_spec.minimum, action_spec.maximum)
        return self._environment.step(action)
