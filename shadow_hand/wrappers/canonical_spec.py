import dm_env
import numpy as np
import tree
from dm_env import specs

from shadow_hand.wrappers import base


class CanonicalSpecWrapper(base.EnvironmentWrapper):
    def __init__(self, environment: dm_env.Environment, clip: bool = False) -> None:
        super().__init__(environment)
        self._action_spec = environment.action_spec()
        self._clip = clip

    def step(self, action) -> dm_env.TimeStep:
        scaled_action = _scale_nested_action(action, self._action_spec, self._clip)
        return self._environment.step(scaled_action)

    def action_spec(self):
        return _convert_spec(self._environment.action_spec())


def _convert_spec(nested_spec):
    """Converts all bounded specs in a nested spec to the canonical scale."""

    def _convert_single_spec(spec):
        if isinstance(spec, specs.BoundedArray):
            return spec.replace(
                minimum=-np.ones(spec.shape), maximum=np.ones(spec.shape)
            )
        else:
            return spec

    return tree.map_structure(_convert_single_spec, nested_spec)


def _scale_nested_action(
    nested_action,
    nested_spec,
    clip: bool,
):
    """Converts a canonical nested action back to the given nested action spec."""

    def _scale_action(action: np.ndarray, spec: specs.Array):
        if isinstance(spec, specs.BoundedArray):
            # Get scale and offset of output action spec.
            scale = spec.maximum - spec.minimum
            offset = spec.minimum

            # Maybe clip.
            if clip:
                action = np.clip(action, -1.0, 1.0)

            # Map action to [0, 1].
            action = 0.5 * (action + 1.0)

            # Map action to [spec.minimum, spec.maximum].
            action *= scale
            action += offset

        return action

    return tree.map_structure(_scale_action, nested_action, nested_spec)
