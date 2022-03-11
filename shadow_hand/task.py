from typing import Optional

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_env import specs

from shadow_hand import effector
from shadow_hand.models.hands import fingered_hand


class Task(composer.Task):
    """Base class for dexterous manipulation tasks.

    This class overrides the `before_step` method by delegating the actuation to the
    `hand_effector`.
    """

    def __init__(
        self,
        arena: composer.Arena,
        hand: fingered_hand.FingeredHand,
        hand_effector: effector.Effector,
    ) -> None:
        self._arena = arena
        self._hand = hand
        self._hand_effector = hand_effector

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        del random_state  # Unused.
        self._hand_effector.set_control(physics, action)

    def after_compile(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del physics, random_state  # Unused.
        self._hand_effector.after_compile(self.root_entity.mjcf_model.root)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._hand_effector.initialize_episode(physics, random_state)

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return self._hand_effector.action_spec(physics)

    @property
    def step_limit(self) -> Optional[int]:
        """The maximum number of steps in an episode."""
        return None
