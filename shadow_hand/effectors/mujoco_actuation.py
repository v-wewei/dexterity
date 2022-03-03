from typing import Sequence

import numpy as np
from dm_control import mjcf
from dm_env import specs

from shadow_hand import effector
from shadow_hand import hints


class MujocoEffector(effector.Effector):
    """A generic effector for multiple MuJoCo actuators."""

    def __init__(
        self,
        actuators: Sequence[hints.MjcfElement],
        prefix: str = "",
    ):
        self._actuators = actuators
        self._prefix = prefix
        self._action_spec = None

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        if self._action_spec is None:
            self._action_spec = create_action_spec(
                physics,
                self._actuators,
                self._prefix,
            )
        return self._action_spec

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        physics.bind(self._actuators).ctrl = command

    @property
    def prefix(self) -> str:
        return self._prefix

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
    control_range = physics.bind(actuators).ctrlrange
    is_limited = physics.bind(actuators).ctrllimited.astype(bool)
    action_min = np.full(num_actuators, fill_value=-np.inf, dtype=np.float32)
    action_max = np.full(num_actuators, fill_value=np.inf, dtype=np.float32)
    action_min[is_limited], action_max[is_limited] = control_range[is_limited].T
    return specs.BoundedArray(
        shape=(num_actuators,),
        dtype=np.float32,
        minimum=action_min,
        maximum=action_max,
        name="\t".join(actuator_names),
    )
