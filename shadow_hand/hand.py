import abc

from dm_control import composer, mjcf


class Hand(abc.ABC, composer.Entity):
    """Composer abstract base class for a hand."""

    @abc.abstractmethod
    def _build(self) -> None:
        """Entity initialization method to be overridden by subclasses."""

    @property
    @abc.abstractmethod
    def mjcf_model(self) -> mjcf.RootElement:
        """Returns the `mjcf.RootElement` object corresponding to the hand."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the hand."""
