from typing import Callable

import numpy as np
from dm_control import mjcf
from dm_env import specs

from dexterity import effector
from dexterity.models.hands import fingered_hand


class EffectorModifier(effector.Effector):
    """Effector wrapper that modfies the control before passing it to a delegate."""

    def __init__(
        self,
        delegate: effector.Effector,
        state_getter: Callable[[mjcf.Physics], np.ndarray],
        modifier_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self._delegate = delegate
        self._state_getter = state_getter
        self._modifier_func = modifier_func

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        cur_state = self._state_getter(physics)
        transformed_command = self._modifier_func(command, cur_state)
        self._delegate.set_control(physics, transformed_command)

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return self._delegate.action_spec(physics)

    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        return self._delegate.after_compile(mjcf_model)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        return self._delegate.initialize_episode(physics, random_state)

    @property
    def prefix(self) -> str:
        return self._delegate.prefix


class RelativeToJointPositions(EffectorModifier):
    def __init__(
        self,
        joint_position_effector: effector.Effector,
        hand: fingered_hand.FingeredHand,
    ):
        def _get_ctrl_state(physics: mjcf.Physics) -> np.ndarray:
            return physics.bind(hand.actuators).ctrl

        def _modify(command: np.ndarray, state: np.ndarray) -> np.ndarray:
            assert len(command) == len(state)
            return command - state

        super().__init__(
            delegate=joint_position_effector,
            state_getter=_get_ctrl_state,
            modifier_func=_modify,
        )
