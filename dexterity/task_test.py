"""Tests for task."""

from typing import List

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_env import specs

from dexterity import effector
from dexterity import task
from dexterity.models import arenas


class DummyEffector(effector.Effector):
    """A dummy effector that actuates nothing."""

    def __init__(self, prefix: str, dof: int) -> None:
        self._prefix = prefix
        self._dof = dof

        self.received_commands: List[np.ndarray] = []

    def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
        pass

    def initialize_episode(
        self,
        physics: mjcf.Physics,
        random_state: np.random.RandomState,
    ) -> None:
        pass

    def action_spec(self, physics):
        del physics  # Unused.
        actuator_names = [(self.prefix + str(i)) for i in range(self._dof)]
        return specs.BoundedArray(
            shape=(self._dof,),
            dtype=np.float32,
            minimum=[-100.0] * self._dof,
            maximum=[100.0] * self._dof,
            name="\t".join(actuator_names),
        )

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        del physics  # Unused.
        self.received_commands.append(command)

    @property
    def prefix(self) -> str:
        return self._prefix


class DummyHand(composer.ModelWrapperEntity):
    """A dummy hand that does nothing."""

    def __init__(self, name: str) -> None:
        super().__init__(mjcf.RootElement(model=name))

    @property
    def name(self) -> str:
        return self.mjcf_model.model


class DummyTask(task.Task):
    """A task with with dummy hands, dummy effectors and 0 reward."""

    def __init__(
        self,
        unique_hand_names: bool = True,
        unique_eff_prefixes: bool = True,
    ) -> None:
        self._null_entity = composer.ModelWrapperEntity(mjcf.RootElement())

        hand1 = DummyHand("hand1")
        hand2 = DummyHand("hand2")
        hand3 = DummyHand("hand3")
        hand4 = DummyHand("hand4" if unique_hand_names else "hand3")

        eff1 = DummyEffector("eff1", 5)
        eff2 = DummyEffector("eff2", 10)
        eff3 = DummyEffector("eff3", 1)
        eff4 = DummyEffector("eff4" if unique_eff_prefixes else "eff3", 0)  # No dof.

        self.dim_shape = (5 + 10 + 1 + 0,)
        self.dims = (5, 10, 1, 0)

        super().__init__(
            arena=arenas.Arena(),
            hands=[hand1, hand2, hand3, hand4],
            hand_effectors=[eff1, eff2, eff3, eff4],
        )

    @property
    def root_entity(self) -> composer.Entity:
        return self._null_entity

    def get_reward(self, physics: mjcf.Physics) -> float:
        del physics  # Unused.
        return 0.0


class TaskTest(parameterized.TestCase):
    def test_raises_value_error_if_effector_prefix_is_not_unique(self) -> None:
        with self.assertRaises(ValueError):
            DummyTask(unique_eff_prefixes=False)

    def test_raises_value_error_if_hand_name_is_not_unique(self) -> None:
        with self.assertRaises(ValueError):
            DummyTask(unique_hand_names=False)

    def test_action_spec(self) -> None:
        task = DummyTask()

        expected_spec = specs.BoundedArray(
            shape=task.dim_shape,
            dtype=np.float32,
            minimum=np.full(shape=task.dim_shape, fill_value=-100.0, dtype=np.float32),
            maximum=np.full(shape=task.dim_shape, fill_value=100.0, dtype=np.float32),
        )

        # Note: specs have `__eq__` defined so we can use `assertEqual`.
        self.assertEqual(task.action_spec(None), expected_spec)

    @parameterized.parameters(
        dict(effector_idx=0, expected_idxs=range(5)),
        dict(effector_idx=1, expected_idxs=range(5, 10 + 5)),
        dict(effector_idx=2, expected_idxs=range(10 + 5, 10 + 5 + 1)),
        dict(effector_idx=3, expected_idxs=[]),
    )
    def test_effector_indices(
        self, effector_idx: int, expected_idxs: List[int]
    ) -> None:
        task = DummyTask()
        mask = task._find_effector_indices(task.hand_effectors[effector_idx], None)
        actual_idxs = np.where(mask)[0]
        np.testing.assert_array_equal(actual_idxs, expected_idxs)


if __name__ == "__main__":
    absltest.main()
