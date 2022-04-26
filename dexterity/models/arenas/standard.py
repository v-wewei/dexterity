from typing import Optional

from dexterity import hints
from dexterity.models.arenas import Arena


class Standard(Arena):
    """An arena with a ground plane."""

    def _build(self, name: Optional[str] = None):
        super()._build(name)

        self._ground = self._mjcf_root.worldbody.add(
            "geom",
            name="ground",
            size="1 1 0.1",
            type="plane",
            friction="0.4 0.005 0.0001",
            solimp="0.95 0.99 0.001",
            solref="0.002 1",
            material="groundplane",
        )

    @property
    def ground(self) -> hints.MjcfElement:
        """The ground plane mjcf element."""
        return self._ground
