"""Tests for dls."""

import mujoco
from absl.testing import absltest
from dm_control import mjcf

from dexterity.controllers import dls
from dexterity.models import hands


class DampedLeastSquaresParametersTest(absltest.TestCase):
    def test_raises_value_error_with_negative_regularization(self) -> None:
        hand = hands.ShadowHandSeriesE()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model.root_model)

        with self.assertRaises(ValueError):
            dls.DampedLeastSquaresParameters(
                model=physics.model,
                regularization_weight=-1,
                object_types=(mujoco.mjtObj.mjOBJ_SITE,),
                object_names=("fftip_site",),
            )


if __name__ == "__main__":
    absltest.main()
