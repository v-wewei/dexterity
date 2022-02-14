"""Tests for geometry_utils."""

import numpy as np
from absl.testing import absltest

from shadow_hand.utils import geometry_utils


class GeometryUtilsTest(absltest.TestCase):
    def test_l2_normalize(self) -> None:
        input_array = np.random.randn(4)
        output_array = geometry_utils.l2_normalize(input_array)
        self.assertTrue(np.isclose(np.linalg.norm(output_array), 1.0))


if __name__ == "__main__":
    absltest.main()
