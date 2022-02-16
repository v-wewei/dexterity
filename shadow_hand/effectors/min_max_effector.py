from typing import Optional

import numpy as np
from dm_control import mjcf
from dm_env import specs

from shadow_hand import effector


class MinMaxEffector(effector.Effector):
    """An effector that only sends min or max commands to a base effector."""

    def __init__(
        self,
        base_effector: effector.Effector,
        min_action: Optional[np.ndarray] = None,
        max_action: Optional[np.ndarray] = None,
    ) -> None:
        self._effector = base_effector
        self._min_act = min_action
        self._max_act = max_action
        self._action_spec = None

    @property
    def prefix(self) -> str:
        return self._effector.prefix

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        if self._action_spec is None:
            self._action_spec = self._effector.action_spec(physics)
            assert self._action_spec is not None

            # Get the minimum action for each DoF. If the user provided no value, use
            # the one from the environment.
            if self._min_act is None:
                self._min_act = self._action_spec.minimum
                if self._min_act.size == 1:
                    self._min_act = np.full(
                        shape=self._action_spec.shape,
                        fill_value=self._min_act,
                        dtype=self._action_spec.dtype,
                    )
            if self._min_act.shape != self._action_spec.shape:
                raise ValueError(
                    f"Expected shape {self._action_spec.shape} for `min_action`, got "
                    f"shape {self._min_act.shape} instead."
                )

            # Get the maximum action for each DoF. If the user provided no value, use
            # the one from the environment.
            if self._max_act is None:
                self._max_act = self._action_spec.maximum
                if self._max_act.size == 1:
                    self._max_act = np.full(
                        shape=self._action_spec.shape,
                        fill_value=self._max_act,
                        dtype=self._action_spec.dtype,
                    )
            if self._max_act.shape != self._action_spec.shape:
                raise ValueError(
                    f"Expected shape {self._action_spec.shape} for `max_action`, got "
                    f"shape {self._max_act.shape} instead."
                )

        return self._action_spec

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        if self._action_spec is None:
            self.action_spec(physics)
        assert self._action_spec is not None

        mid_point = 0.5 * (self._action_spec.minimum + self._action_spec.maximum)
        min_idxs = command <= mid_point
        max_idxs = command > mid_point

        new_cmd = np.zeros(shape=self._action_spec.shape, dtype=self._action_spec.dtype)
        new_cmd[min_idxs] = self._min_act[min_idxs]
        new_cmd[max_idxs] = self._max_act[max_idxs]

        self._effector.set_control(physics, new_cmd)

    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        self._effector.after_compile(mjcf_model)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._effector.initialize_episode(physics, random_state)
