from typing import List

import numpy as np
from dm_control import composer
from dm_control import mjcf

from dexterity.hints import MjcfElement
from dexterity.models.hands import adroit_hand_constants as consts
from dexterity.models.hands import fingered_hand
from dexterity.utils import mujoco_utils


class AdroitHand(fingered_hand.FingeredHand):
    def _build(
        self,
        name: str = "adroit_hand",
    ) -> None:
        self._mjcf_root = mjcf.from_path(str(consts.ADROIT_HAND_E_XML))
        self._mjcf_root.model = name

        self._parse_mjcf_elements()

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.

        # Apply gravity compensation.
        mujoco_utils.compensate_gravity(physics, self.mjcf_model.find_all("body"))

    # ================= #
    # Accessors.
    # ================= #

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @property
    def joints(self) -> List[MjcfElement]:
        """List of joint elements belonging to the hand."""
        return self._joints

    @property
    def actuators(self) -> List[MjcfElement]:
        """List of actuator elements belonging to the hand."""
        return self._actuators

    @property
    def tendons(self) -> List[MjcfElement]:
        """List of tendon elements belonging to the hand."""
        return self._tendons

    @property
    def joint_torque_sensors(self) -> List[MjcfElement]:
        """List of joint torque sensor elements belonging to the hand."""
        return self._joint_torque_sensors

    @property
    def fingertip_sites(self) -> List[MjcfElement]:
        """List of fingertip site elements belonging to the hand."""
        return self._fingertip_sites

    # ================= #
    # Public methods.
    # ================= #

    def zero_joint_positions(self) -> np.ndarray:
        return np.zeros(self._num_joints, dtype=float)

    def zero_control(self) -> np.ndarray:
        return np.zeros(self._num_actuators, dtype=float)

    @classmethod
    def control_to_joint_positions(cls, control: np.ndarray) -> np.ndarray:
        return control

    @classmethod
    def joint_positions_to_control(cls, qpos: np.ndarray) -> np.ndarray:
        return qpos

    def set_joint_angles(self, physics: mjcf.Physics, joint_angles: np.ndarray) -> None:
        physics.bind(self._joints).qpos = joint_angles

    def sample_joint_angles(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> np.ndarray:
        return random_state.uniform(*physics.bind(self._joints).range.T)

    # ================= #
    # Private methods.
    # ================= #

    def _parse_mjcf_elements(self) -> None:
        """Parses MJCF elements that will be exposed as attributes."""
        self._joints: List[mjcf.Element] = self._mjcf_root.find_all("joint")
        self._num_joints = len(self._joints)

        self._tendons: List[mjcf.Element] = self._mjcf_root.find_all("tendon")
        self._num_tendons = len(self._tendons)

        self._actuators: List[mjcf.Element] = self._mjcf_root.find_all("actuator")
        self._num_actuators = len(self._actuators)

        self._fingertip_sites: List[mjcf.Element] = [
            elem
            for elem in self._mjcf_root.find_all("site")
            if elem.name.endswith("tip") and elem.name.startswith("S")
        ]

        self._joint_torque_sensors: List[mjcf.Element] = []
        for joint_elem in self._joints:
            site_elem = joint_elem.parent.add(
                "site",
                name=joint_elem.name + "_site",
                # NOTE(kevin): The kwargs below are for visualization purposes.
                size="0.001 0.001 0.001",
                type="box",
                rgba="0 1 0 1",
                group=composer.SENSOR_SITES_GROUP,
            )
            # Create a 3-axis torque sensor.
            torque_sensor_elem = joint_elem.root.sensor.add(
                "torque",
                site=site_elem,
                name=joint_elem.name + "_torque",
            )
            self._joint_torque_sensors.append(torque_sensor_elem)
