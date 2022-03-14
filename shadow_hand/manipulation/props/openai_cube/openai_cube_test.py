"""Tests for the OpenAI cube prop."""

from typing import Sequence, Union

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf

from shadow_hand.manipulation.props import OpenAICube


class OpenAICubeTest(parameterized.TestCase):
    @parameterized.parameters(
        [
            (0.1,),
            ((0.1, 0.1, 0.1),),
        ]
    )
    def test_can_compile_and_step_model(
        self, size: Union[float, Sequence[float]]
    ) -> None:
        prop = OpenAICube(size=size)
        physics = mjcf.Physics.from_mjcf_model(prop.mjcf_model)
        physics.step()

    def test_raises_error_if_size_wrong_len(self) -> None:
        with self.assertRaises(ValueError):
            OpenAICube(size=(1, 2, 3, 4, 5))


if __name__ == "__main__":
    absltest.main()
