from typing import Sequence, Tuple

import numpy as np
from dm_control import mjcf
from dm_env import specs

from shadow_hand import effector, hints


class MujocoEffector(effector.Effector):
    """A generic effector for various MuJoCo actuators."""

    def __init__(
        self,
        actuators: Sequence[hints.MjcfElement],
        prefix: str = "",
    ) -> None:
        self._actuators = actuators
        self._prefix = prefix
        self._action_spec = None

    @property
    def prefix(self) -> str:
        return self._prefix

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        if self._action_spec is None:
            self._action_spec = create_action_spec(
                physics,
                self._actuators,
                self._prefix,
            )
        return self._action_spec

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        self.action_spec(physics).validate(command)
        physics.bind(self._actuators).ctrl = command

    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        pass

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        pass


def create_action_spec(
    physics: mjcf.Physics,
    actuators: Sequence[hints.MjcfElement],
    prefix: str = "",
) -> specs.BoundedArray:
    """Creates an action range for the given actuators."""
    num_actuators = len(actuators)
    actuator_names = [f"{prefix}{i}" for i in range(num_actuators)]
    action_min, action_max = _action_range_from_actuators(physics, actuators)
    return specs.BoundedArray(
        shape=(num_actuators,),
        dtype=np.float32,
        minimum=action_min,
        maximum=action_max,
        name="\t".join(actuator_names),
    )


def _action_range_from_actuators(
    physics: mjcf.Physics, actuators: Sequence[hints.MjcfElement]
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the action range for the given actuators."""
    num_actions = len(actuators)
    control_range = physics.bind(actuators).ctrlrange
    is_limited = physics.bind(actuators).ctrllimited.astype(bool)
    minima = np.full(num_actions, fill_value=-np.inf, dtype=np.float32)
    maxima = np.full(num_actions, fill_value=np.inf, dtype=np.float32)
    minima[is_limited], maxima[is_limited] = control_range[is_limited].T
    return minima, maxima
