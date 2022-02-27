"""Tests for inhand_manipulation loader."""

from absl.testing import absltest

from shadow_hand.tasks import inhand_manipulation


class LoaderConstantsTest(absltest.TestCase):
    def testConstants(self) -> None:
        self.assertNotEmpty(inhand_manipulation.ALL)


if __name__ == "__main__":
    absltest.main()
