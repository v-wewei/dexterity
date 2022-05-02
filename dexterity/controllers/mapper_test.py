"""Tests for dls."""

import mujoco
from absl.testing import absltest
from dm_control import mjcf

from dexterity.controllers import mapper
from dexterity.models import hands


class ParametersTest(absltest.TestCase):
    def test_raises_value_error_with_wrong_type_right_name(self) -> None:
        hand = hands.ShadowHandSeriesE()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model.root_model)

        with self.assertRaises(ValueError):
            mapper.Parameters(
                model=physics.model,
                object_types=(mujoco.mjtObj.mjOBJ_JOINT,),
                object_names=("fftip_site",),
            )

    def test_raises_value_error_with_wrong_name_right_type(self) -> None:
        hand = hands.ShadowHandSeriesE()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model.root_model)

        with self.assertRaises(ValueError):
            mapper.Parameters(
                model=physics.model,
                object_types=(mujoco.mjtObj.mjOBJ_SITE,),
                object_names=("nonexistent_site_name",),
            )


if __name__ == "__main__":
    absltest.main()
