"""Tests for the empty."""

from absl.testing import absltest
from dm_control import mjcf

from shadow_hand.manipulation.arenas import empty


class EmptyTest(absltest.TestCase):
    def test_can_compile_and_step_model(self) -> None:
        arena = empty.Empty()
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        physics.step()


if __name__ == "__main__":
    absltest.main()
