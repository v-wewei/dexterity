"""Tests for the OpenAI cube prop."""

from absl.testing import absltest
from dm_control import mjcf

from shadow_hand.manipulation.props import OpenAICube


class OpenAICubeTest(absltest.TestCase):
    def test_can_compile_and_step_model(self) -> None:
        prop = OpenAICube(size=0.01)
        physics = mjcf.Physics.from_mjcf_model(prop.mjcf_model)
        physics.step()


if __name__ == "__main__":
    absltest.main()
