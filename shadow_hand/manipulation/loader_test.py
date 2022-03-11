"""Tests for manipulation loader."""

from absl.testing import absltest

from shadow_hand import manipulation


class LoaderConstantsTest(absltest.TestCase):
    def testConstants(self) -> None:
        self.assertNotEmpty(manipulation.ALL)


if __name__ == "__main__":
    absltest.main()
