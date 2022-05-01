from typing import Optional

import numpy as np
from dm_control import mjcf

from dexterity import effector
from dexterity.effectors.wrappers import base


class PreviousAction(base.Wrapper):
    """Wraps an effector and stores the most recent action."""

    def __init__(self, effector: effector.Effector) -> None:
        super().__init__(effector)

        self._previous_action: Optional[np.ndarray] = None

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._previous_action = np.zeros_like(self.action_spec(physics).minimum)

        super().initialize_episode(physics, random_state)

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        super().set_control(physics, command)

        # Note: Slower than `self._previous_action = command`, but safer in case command
        # is mutated elsewhere.
        self._previous_action[:] = command  # type: ignore

    @property
    def previous_action(self) -> Optional[np.ndarray]:
        return self._previous_action
