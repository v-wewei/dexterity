import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_env import specs

from shadow_hand import effector
from shadow_hand import hand


class Task(composer.Task):
    def __init__(
        self,
        arena: composer.Arena,
        hand: hand.Hand,
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
        del random_state  # Unused.
        self._hand_effector.after_compile(physics)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._hand_effector.initialize_episode(physics, random_state)

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return self._hand_effector.action_spec(physics)
