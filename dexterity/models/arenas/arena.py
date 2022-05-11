from typing import Optional

from dm_control import composer
from dm_control import mjcf

from dexterity import _SRC_ROOT
from dexterity import hints

_ARENA_XML = _SRC_ROOT / "models" / "arenas" / "arena.xml"


class Arena(composer.Arena):
    """Modified composer Arena."""

    def _build(self, name: Optional[str] = None):
        super()._build(name)

        self._mjcf_root.include_copy(
            mjcf.from_path(_ARENA_XML), override_attributes=True
        )

        # Remove the default visual headlight setting in `dm_control.Arena`.
        self._mjcf_root.visual.headlight.remove("headlight")

        # Remove any default lights.
        for light_elem in self._mjcf_root.worldbody.find_all("light"):
            light_elem.remove()

        # Add our own custom lights.
        self._mjcf_root.worldbody.add(
            "light",
            pos=(0, 0, 1.5),
            dir=(0, 0, -1),
            diffuse=(0.7, 0.7, 0.7),
            specular=(0.3, 0.3, 0.3),
            directional="true",
            castshadow="true",
        )

    def attach_offset(
        self,
        entity: composer.Entity,
        position: Optional[hints.FloatArray] = None,
        quaternion: Optional[hints.FloatArray] = None,
        attach_site: Optional[hints.MjcfAttachmentFrame] = None,
    ):
        frame = self.attach(entity, attach_site=attach_site)
        if position is not None:
            if len(position) != 3:
                raise ValueError("Position must be a sequence of length 3.")
            frame.pos = position
        if quaternion is not None:
            if len(quaternion) != 4:
                raise ValueError("Quaternion must be a sequence of length 4.")
            frame.quat = quaternion
        return frame

    def add_mocap(
        self,
        entity: composer.Entity,
        position: Optional[hints.FloatArray] = None,
        quaternion: Optional[hints.FloatArray] = None,
        visible: bool = False,
        color: hints.RgbaColor = (0.9, 0.5, 0.5, 1.0),
        name: str = "mocap",
    ) -> hints.MjcfElement:
        # Add mocap body.
        mocap_body = self.mjcf_model.worldbody.add(
            "body",
            name=name,
            mocap="true",
            pos=position,
            quat=quaternion,
        )

        # Give it a square shape.
        if visible:
            mocap_body.add(
                "geom",
                type="box",
                group="1",
                size="0.02 0.02 0.02",
                contype="0",
                conaffinity="0",
                rgba=color,
            )

        # Make root body of entity's pose same as mocap pose.
        root_body = entity.mjcf_model.find_all("body")[0]
        root_body.pos = position
        root_body.quat = quaternion

        # Add the entity to the arena as a free body.
        self.add_free_entity(entity)

        # Add a weld constraint between the mocap body and the root body.
        self.mjcf_model.equality.add(
            "weld",
            body1=name,
            body2=root_body.full_identifier,
            solref="0.01 1",
            solimp=".9 .9 0.01",
        )

        return mocap_body
