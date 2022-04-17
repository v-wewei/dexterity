"""Tests for the standard."""

from absl.testing import absltest
from dm_control import mjcf

from dexterity.manipulation.arenas import standard


class StandardTest(absltest.TestCase):
    def test_can_compile_and_step_model(self) -> None:
        arena = standard.Standard()
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        physics.step()


if __name__ == "__main__":
    absltest.main()
