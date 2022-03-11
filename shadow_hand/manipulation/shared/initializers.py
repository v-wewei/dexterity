"""Shared initializers for the hand."""

from typing import Sequence

import numpy as np
from dm_control import composer
from dm_control import mjcf

from shadow_hand import hints
from shadow_hand.models.hands import fingered_hand

_REJECTION_SAMPLING_FAILED = (
    "Failed to find a collision-free initial configuration for the fingertips after "
    "{max_rejection_samples} randomly sampled joint configurations."
)


class FingertipPositionPlacer(composer.Initializer):
    """An initializer that sets target site positions for the fingertips of a hand.

    This initializer works backwards by sampling a joint configuration for the entire
    hand, then querying the fingertip positions using forward kinematics. This ensures
    that the fingertip sites are always reachable.
    """

    def __init__(
        self,
        target_sites: Sequence[hints.MjcfElement],
        hand: fingered_hand.FingeredHand,
        ignore_self_collisions: bool = False,
        max_rejection_samples: int = 100,
    ) -> None:
        """Constructor.

        Args:
            target_sites: The target fingertip sites to place.
            hand: An instance of `fingered_hand.FingeredHand`.
            ignore_self_collisions: If True, self-collisions are ignored, i.e.,
                rejection sampling is disabled.
            max_rejection_samples: The maximum number of joint configurations to sample
                while attempting to find a collision-free configuration.
        """
        super().__init__()

        self._target_sites = target_sites
        self._hand = hand
        self._ignore_self_collisions = ignore_self_collisions
        self._max_rejection_samples = max_rejection_samples

    def _has_self_collisions(self, physics: mjcf.Physics) -> bool:
        """Returns True if the hand is in a self-collision state."""
        mjcf_root = self._hand.mjcf_model.root_model
        hand_model = self._hand.mjcf_model
        all_geoms = mjcf_root.find_all("geom")
        for contact in physics.data.contact:
            geom_1 = all_geoms[contact.geom1]
            geom_2 = all_geoms[contact.geom2]
            # Ignore contacts with positive distance (i.e., not actually touching).
            if contact.dist > 0:
                continue
            # Check for hand-hand collisions.
            if geom_1.root is hand_model and geom_2.root is hand_model:
                return True
        return False

    def __call__(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Sets the position of the fingertip target sites.

        Raises:
            RuntimeError: If a collision-free configuration of the fingertips could not
                be found after `max_rejection_samples` randomly sampled joint
                configurations.
        """

        initial_qpos = physics.bind(self._hand.joints).qpos.copy()
        fingertip_pos = None

        for _ in range(self._max_rejection_samples):
            # Sample a random joint configuration.
            qpos = random_state.uniform(
                physics.bind(self._hand.joints).range[:, 0],
                physics.bind(self._hand.joints).range[:, 1],
            )
            physics.bind(self._hand.joints).qpos = qpos

            physics.forward()

            # `or` is a short-circuit operator in python, which means collision checking
            # is only performed if ignore_self_collisions evaluates to False.
            # See: https://docs.python.org/3/library/stdtypes.html#boolean-operations-and-or-not
            if self._ignore_self_collisions or not self._has_self_collisions(physics):
                fingertip_pos = physics.bind(self._hand.fingertip_sites).xpos.copy()
                break

        # Restore the initial joint configuration.
        physics.bind(self._hand.joints).qpos = initial_qpos

        if fingertip_pos is None:
            raise RuntimeError(
                _REJECTION_SAMPLING_FAILED.format(
                    max_rejection_samples=self._max_rejection_samples
                )
            )
        else:
            physics.bind(self._target_sites).pos = fingertip_pos
