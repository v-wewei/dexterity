from typing import Optional, Sequence

from dm_control import composer

from shadow_hand import hints


class Arena(composer.Arena):
    """Standard composer arena with added functionality."""

    def attach_offset(
        self,
        entity: composer.Entity,
        position: Optional[Sequence[float]] = None,
        quaternion: Optional[Sequence[float]] = None,
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
