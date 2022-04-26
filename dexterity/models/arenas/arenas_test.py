"""Tests for the arenas."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf

from dexterity.models import arenas


class ArenasTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("Arena", arenas.Arena),
        ("Standard", arenas.Standard),
    )
    def test_can_compile_and_step_model(self, arena_cls) -> None:
        arena = arena_cls()
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        physics.step()


if __name__ == "__main__":
    absltest.main()
