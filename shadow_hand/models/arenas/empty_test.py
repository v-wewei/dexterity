"""Tests for empty."""

from absl.testing import absltest
from dm_control import mjcf

from shadow_hand.models.arenas import empty


class TestEmpty(absltest.TestCase):
    def test_can_compile_and_step_model(self) -> None:
        arena = empty.Arena()
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        physics.step()


if __name__ == "__main__":
    absltest.main()
