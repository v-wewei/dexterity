import abc
from typing import List

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.transformations import transformations as tr

from dexterity.hints import MjcfElement
from dexterity.utils import mujoco_utils


class FingeredHand(abc.ABC, composer.Entity):
    """Base composer class for a multi-fingered hand."""

    def _build_observables(self) -> composer.Observables:
        return RobotHandObservables(self)

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

    @property
    @abc.abstractmethod
    def actuators(self) -> List[MjcfElement]:
        """List of actuator elements belonging to the hand."""

    @property
    @abc.abstractmethod
    def joints(self) -> List[MjcfElement]:
        """List of joint elements belonging to the hand."""

    @property
    @abc.abstractmethod
    def fingertip_sites(self) -> List[MjcfElement]:
        """List of fingertip site elements belonging to the hand."""

    @property
    @abc.abstractmethod
    def joint_torque_sensors(self) -> List[MjcfElement]:
        """List of joint torque sensor elements belonging to the hand."""

    @abc.abstractmethod
    def set_joint_angles(self, physics: mjcf.Physics, joint_angles: np.ndarray) -> None:
        """Sets the joints of the hand to a given configuration."""

    @abc.abstractmethod
    def sample_joint_angles(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> np.ndarray:
        """Samples a random joint configuration for the hand."""


class RobotHandObservables(composer.Observables):
    """Observables for a fingered hand."""

    _entity: FingeredHand

    # shape: (1, 24)
    @composer.observable
    def joint_positions(self) -> observable.MJCFFeature:
        return observable.MJCFFeature(kind="qpos", mjcf_element=self._entity.joints)

    # shape: (1, 48)
    @composer.observable
    def joint_positions_sin_cos(self) -> observable.MJCFFeature:
        def _get_joint_pos_sin_cos(physics: mjcf.Physics) -> np.ndarray:
            qpos = physics.bind(self._entity.joints).qpos
            qpos_sin = np.sin(qpos)
            qpos_cos = np.cos(qpos)
            return np.concatenate([qpos_sin, qpos_cos])

        return observable.Generic(raw_observation_callable=_get_joint_pos_sin_cos)

    # shape: (1, 24)
    @composer.observable
    def joint_velocities(self) -> observable.MJCFFeature:
        return observable.MJCFFeature(kind="qvel", mjcf_element=self._entity.joints)

    # shape: (1, 24)
    @composer.observable
    def joint_torques(self) -> observable.Generic:
        def _get_joint_torques(physics: mjcf.Physics) -> np.ndarray:
            torques = physics.bind(self._entity.joint_torque_sensors).sensordata
            joint_axes = physics.bind(self._entity.joints).axis
            joint_torques = np.einsum("ij,ij->i", torques.reshape(-1, 3), joint_axes)
            return joint_torques

        return observable.Generic(raw_observation_callable=_get_joint_torques)

    # shape: (1, 15)
    @composer.observable
    def fingertip_positions(self) -> observable.Generic:
        def _get_fingertip_positions(physics: mjcf.Physics) -> np.ndarray:
            return physics.bind(self._entity.fingertip_sites).xpos.ravel()

        return observable.Generic(raw_observation_callable=_get_fingertip_positions)

    # shape: (1, 20)
    @composer.observable
    def fingertip_orientations(self) -> observable.Generic:
        def _get_fingertip_orientations(physics: mjcf.Physics) -> np.ndarray:
            xmats = physics.bind(self._entity.fingertip_sites).xmat.reshape(-1, 3, 3)
            quats = [tr.mat_to_quat(xmat) for xmat in xmats]
            return np.concatenate(quats)

        return observable.Generic(raw_observation_callable=_get_fingertip_orientations)

    # shape: (1, 15)
    @composer.observable
    def fingertip_linear_velocities(self) -> observable.Generic:
        def _get_fingertip_linear_velocities(physics: mjcf.Physics) -> np.ndarray:
            linear_velocities = []
            for fingertip_site in self._entity.fingertip_sites:
                lin_vel = mujoco_utils.get_site_velocity(physics, fingertip_site)
                linear_velocities.append(lin_vel[:3])
            return np.concatenate(linear_velocities)

        return observable.Generic(
            raw_observation_callable=_get_fingertip_linear_velocities
        )

    # shape: (1, 15)
    @composer.observable
    def fingertip_angular_velocities(self) -> observable.Generic:
        def _get_fingertip_angular_velocities(physics: mjcf.Physics) -> np.ndarray:
            angular_velocities = []
            for fingertip_site in self._entity.fingertip_sites:
                ang_vel = mujoco_utils.get_site_velocity(physics, fingertip_site)
                angular_velocities.append(ang_vel[3:])
            return np.concatenate(angular_velocities)

        return observable.Generic(
            raw_observation_callable=_get_fingertip_angular_velocities
        )
