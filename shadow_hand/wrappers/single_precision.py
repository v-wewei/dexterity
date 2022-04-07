import dm_env
import numpy as np
import tree
from dm_env import specs

from shadow_hand.wrappers import base


class SinglePrecisionWrapper(base.EnvironmentWrapper):
    def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        return timestep._replace(
            reward=_convert_value(timestep.reward),
            discount=_convert_value(timestep.discount),
            observation=_convert_value(timestep.observation),
        )

    def step(self, action) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.step(action))

    def reset(self) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.reset())

    def action_spec(self):
        return _convert_spec(self._environment.action_spec())

    def discount_spec(self):
        return _convert_spec(self._environment.discount_spec())

    def observation_spec(self):
        return _convert_spec(self._environment.observation_spec())

    def reward_spec(self):
        return _convert_spec(self._environment.reward_spec())


def _convert_spec(nested_spec):
    """Converts a nested spec."""

    def _convert_single_spec(spec: specs.Array):
        if spec.dtype == "O":
            # Pass StringArray objects through unmodified.
            return spec
        if np.issubdtype(spec.dtype, np.float64):
            dtype = np.float32
        elif np.issubdtype(spec.dtype, np.int64):
            dtype = np.int32  # type: ignore
        else:
            dtype = spec.dtype
        return spec.replace(dtype=dtype)

    return tree.map_structure(_convert_single_spec, nested_spec)


def _convert_value(nested_value):
    def _convert_single_value(value):
        if value is not None:
            value = np.array(value, copy=False)
            if np.issubdtype(value.dtype, np.float64):
                value = np.array(value, copy=False, dtype=np.float32)
            elif np.issubdtype(value.dtype, np.int64):
                value = np.array(value, copy=False, dtype=np.int32)
        return value

    return tree.map_structure(_convert_single_value, nested_value)
