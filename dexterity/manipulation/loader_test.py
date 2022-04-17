"""Tests for manipulation loader."""

from absl.testing import absltest

from dexterity import manipulation


class LoaderConstantsTest(absltest.TestCase):
    def testConstants(self) -> None:
        self.assertNotEmpty(manipulation.ALL_TASKS)
        self.assertNotEmpty(manipulation.TASKS_BY_DOMAIN)


if __name__ == "__main__":
    absltest.main()
