from typing import Optional

from shadow_hand import arena


class Empty(arena.Arena):
    """An empty arena."""

    def _build(self, name: Optional[str] = None) -> None:
        super()._build(name=name)

        # Remove the default visual headlight setting in `dm_control.Arena`.
        self._mjcf_root.visual.remove("headlight")

        # Add visual assets.
        self.mjcf_model.asset.add(
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1=(0.4, 0.6, 0.8),
            rgb2=(0.0, 0.0, 0.0),
            width=100,
            height=100,
        )

        # Add lighting.
        self.mjcf_model.worldbody.add(
            "light",
            pos=(0, 0, 1.5),
            dir=(0, 0, -1),
            diffuse=(0.7, 0.7, 0.7),
            specular=(0.3, 0.3, 0.3),
            directional="true",
            castshadow="true",
        )

        # Always initialize the free camera so that it points at the origin.
        self.mjcf_model.statistic.center = (0.0, 0.0, 0.0)
