"""Shared initializers for the hand."""

from typing import Optional, Sequence

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.initializers import utils

from shadow_hand import hints
from shadow_hand.models.hands import fingered_hand
from shadow_hand.utils import mujoco_collisions
from shadow_hand.utils import mujoco_utils

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
        self._qpos = None

    def _has_self_collisions(self, physics: mjcf.Physics) -> bool:
        """Returns True if the hand is in a self-collision state."""
        return mujoco_collisions.has_self_collision(physics, self._hand.name)

    def __call__(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Sets the position of the fingertip target sites.

        Raises:
            RuntimeError: If a collision-free configuration of the fingertips could not
                be found after `max_rejection_samples` randomly sampled joint
                configurations.
        """
        joint_binding = physics.bind(self._hand.joints)
        actuator_binding = physics.bind(self._hand.actuators)

        initial_qpos = joint_binding.qpos.copy()
        initial_ctrl = actuator_binding.ctrl.copy()
        fingertip_pos = None

        # # Apply gravity compensation.
        mujoco_utils.compensate_gravity(physics, self._hand.mjcf_model.find_all("body"))

        for _ in range(self._max_rejection_samples):
            # Sample a random joint configuration.
            qpos_desired = self._hand.sample_joint_angles(physics, random_state)
            self._hand.set_joint_angles(physics, qpos_desired)
            physics.forward()

            if self._ignore_self_collisions or not self._has_self_collisions(physics):
                ctrl_desired = self._hand.joint_positions_to_control(qpos_desired)

                self._hand.set_joint_angles(physics, initial_qpos)
                actuator_binding.ctrl[:] = ctrl_desired

                original_time = physics.data.time
                hand_isolator = utils.JointStaticIsolator(physics, self._hand.joints)
                qpos_prev = None
                while True:
                    with hand_isolator:
                        physics.step()
                    qpos = joint_binding.qpos.copy()
                    if qpos_prev is not None:
                        if np.all(np.abs(qpos_prev - qpos) <= 1e-3):
                            break
                    qpos_prev = qpos
                physics.data.time = original_time

                # At this point, the fingers could have collided and gotten stuck in a
                # stalemate. Thus, we check again and discard the solution if any
                # contacts were detected.
                if self._has_self_collisions(physics):
                    continue

                fingertip_pos = physics.bind(self._hand.fingertip_sites).xpos.copy()
                self._qpos = qpos_desired
                break

        # Restore the initial joint configuration and ctrl.
        self._hand.set_joint_angles(physics, initial_qpos)
        actuator_binding.ctrl[:] = initial_ctrl

        if fingertip_pos is None:
            raise RuntimeError(
                _REJECTION_SAMPLING_FAILED.format(
                    max_rejection_samples=self._max_rejection_samples
                )
            )
        else:
            physics.bind(self._target_sites).pos = fingertip_pos

    @property
    def qpos(self) -> Optional[np.ndarray]:
        """The joint configuration used to place the target sites."""
        return self._qpos
