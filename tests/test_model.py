from absl.testing import absltest
from dm_control import mjcf

from shadow_hand import shadow_hand_e_constants as consts


class ShadowHandTest(absltest.TestCase):
    def test_physics_step(self) -> None:
        pass


if __name__ == "__main__":
    absltest.main()
