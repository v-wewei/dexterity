"""Tests for the arenas."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
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


class StandardArenaTest(absltest.TestCase):
    def test_raises_value_error_if_position_or_quaternion_wrong_length(self) -> None:
        arena = arenas.Standard()
        entity = composer.ModelWrapperEntity(mjcf.RootElement(model="null_entity"))

        with self.subTest("position_wrong_length"):
            with self.assertRaises(ValueError):
                arena.attach_offset(entity=entity, position=[0])

        entity.detach()

        with self.subTest("quaternion_wrong_length"):
            with self.assertRaises(ValueError):
                arena.attach_offset(entity=entity, quaternion=[0])


if __name__ == "__main__":
    absltest.main()
