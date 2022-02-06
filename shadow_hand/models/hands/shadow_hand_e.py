import dataclasses
from typing import List

import numpy as np
from dm_control import mjcf

from shadow_hand import hand
from shadow_hand.hints import MjcfElement
from shadow_hand.models.hands import shadow_hand_e_constants as consts


# NOTE(kevin): There's a damping parameter at the <joint> level, which means we in fact
# have a PD-based position controller under the hood. So `kp` here is the proportional
# gain and the damping parameter is related to the derivative gain.
@dataclasses.dataclass
class _ActuatorParams:
    # Position feedback gain.
    kp: float = 1.0


# NOTE(kevin): The values of the constants below are not really tuned. At the moment,
# I just loaded the model in the MuJoCo viewer and set the values manually to get
# visually reasonable results.
_ACTUATOR_PARAMS = {
    consts.Actuation.POSITION: {
        # Wrist.
        consts.Actuators.A_WRJ1: _ActuatorParams(kp=20.0),
        consts.Actuators.A_WRJ0: _ActuatorParams(kp=20.0),
        # First finger.
        consts.Actuators.A_FFJ3: _ActuatorParams(kp=0.6),
        consts.Actuators.A_FFJ2: _ActuatorParams(kp=0.6),
        consts.Actuators.A_FFJ1: _ActuatorParams(kp=0.6),
        # Middle finger.
        consts.Actuators.A_MFJ3: _ActuatorParams(kp=0.6),
        consts.Actuators.A_MFJ2: _ActuatorParams(kp=0.6),
        consts.Actuators.A_MFJ1: _ActuatorParams(kp=0.6),
        # Ring finger.
        consts.Actuators.A_RFJ3: _ActuatorParams(kp=0.6),
        consts.Actuators.A_RFJ2: _ActuatorParams(kp=0.6),
        consts.Actuators.A_RFJ1: _ActuatorParams(kp=0.6),
        # Little finger.
        consts.Actuators.A_LFJ4: _ActuatorParams(kp=0.6),
        consts.Actuators.A_LFJ3: _ActuatorParams(kp=0.6),
        consts.Actuators.A_LFJ2: _ActuatorParams(kp=0.6),
        consts.Actuators.A_LFJ1: _ActuatorParams(kp=0.6),
        # Thumb.
        consts.Actuators.A_THJ4: _ActuatorParams(kp=3.0),
        consts.Actuators.A_THJ3: _ActuatorParams(kp=3.0),
        consts.Actuators.A_THJ2: _ActuatorParams(kp=3.0),
        consts.Actuators.A_THJ1: _ActuatorParams(kp=3.0),
        consts.Actuators.A_THJ0: _ActuatorParams(kp=3.0),
    }
}


class ShadowHandSeriesE(hand.Hand):
    """Shadow Dexterous Hand E Series."""

    def _build(
        self,
        name: str = "shadow_hand_e",
        actuation: consts.Actuation = consts.Actuation.POSITION,
        randomize_color: bool = False,
    ) -> None:
        """Initializes the hand.

        Args:
            name: The name of the hand. Used as a prefix in the MJCF name attributes.
            actuation: Instance of `shadow_hand_e_constants.Actuation` specifying which
                actuation method to use.
            randomize_color: Whether to randomize the color of the hand.
        """
        self._mjcf_root = mjcf.from_path(str(consts.SHADOW_HAND_E_XML))

        self._mjcf_root.model = name
        self._actuation = actuation
        self._randomize_color = randomize_color

        self._add_mjcf_elements()
        self._add_tendons()
        self._add_actuators()
        self._add_sensors()

        if self._randomize_color:
            self._color_hand()

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

    # ================= #
    # Public methods.
    # ================= #

    @classmethod
    def zero_joint_positions(cls) -> np.ndarray:
        return np.zeros(consts.NUM_JOINTS, dtype=float)

    @classmethod
    def zero_control(cls) -> np.ndarray:
        return np.zeros(consts.NUM_ACTUATORS, dtype=float)

    @classmethod
    def control_to_joint_positions(cls, control: np.ndarray) -> np.ndarray:
        """Maps a 20-D position control command to a 24-D joint position command.

        The control commands for the coupled joints are evenly split amongst them.
        """
        if control.shape != (consts.NUM_ACTUATORS,):
            raise ValueError(
                f"Expected control of shape ({consts.NUM_ACTUATORS}), got {control.shape}"
            )
        return consts.CONTROL_TO_POSITION @ control

    @classmethod
    def joint_positions_to_control(cls, qpos: np.ndarray) -> np.ndarray:
        """Maps a 24-D joint position command to a 20-D control command.

        The position commands for the coupled joints are summed up to form the control
        for their corresponding actuator.
        """
        if qpos.shape != (consts.NUM_JOINTS,):
            raise ValueError(
                f"Expected qpos of shape ({consts.NUM_JOINTS}), got {qpos.shape}"
            )
        return consts.POSITION_TO_CONTROL @ qpos

    def actuator_ctrl_range(self, physics: mjcf.Physics) -> np.ndarray:
        """Returns lower and upper bounds on the actuator controls.

        Args:
            physics: An `mjcf.Physics` instance.

        Returns:
            A (20, 2) ndarray containing (lower, upper) control bounds.
        """
        # These are queried from the mjcf instead of the hard-coded constants. This is
        # to account for any possible runtime randomizations.
        return np.array(physics.bind(self._actuators).ctrlrange, copy=True)

    def joint_limits(self, physics: mjcf.Physics) -> np.ndarray:
        """Returns lower and upper bounds on the joint positions.

        Args:
            physics: An `mjcf.Physics` instance.

        Returns:
            A (24, 2) ndarray containing (lower, upper) position bounds.
        """
        # These are queried from the mjcf instead of the hard-coded constants. This is
        # to account for any possible runtime randomizations.
        return np.array(physics.bind(self._joints).range, copy=True)

    def clip_position_control(
        self, physics: mjcf.Physics, control: np.ndarray
    ) -> np.ndarray:
        """Clips the position control vector to the supported range.

        Args:
            physics: A `mujoco.Physics` instance.
            control: The position control vector, of shape (20,).
        """
        bounds = self.actuator_ctrl_range(physics)
        return np.clip(
            a=control,
            a_min=bounds[:, 0],
            a_max=bounds[:, 1],
        )

    def set_position_control(self, physics: mjcf.Physics, control: np.ndarray) -> None:
        """Sets the positions of the joints to the given control command.

        Each coordinate in the control vector is the desired joint angle, or sum of
        joint angles for coupled joints.

        Args:
            physics: A `mujoco.Physics` instance.
            control: The position control vector, of shape (20,).
        """
        # TODO(kevin): Should we clip instead?
        if not self.is_position_control_valid(physics, control):
            raise ValueError("Position control command is invalid.")

        physics_actuators = physics.bind(self._actuators)
        physics_actuators.ctrl[:] = control

    def is_position_control_valid(
        self, physics: mjcf.Physics, control: np.ndarray
    ) -> bool:
        """Returns True if the given position control command is valid."""
        ctrl_bounds = self.actuator_ctrl_range(physics)
        shape_cond = control.shape == (consts.NUM_ACTUATORS,)
        lower_cond = bool(np.all(control >= ctrl_bounds[:, 0] - consts.EPSILON))
        upper_cond = bool(np.all(control <= ctrl_bounds[:, 1] + consts.EPSILON))
        return shape_cond and lower_cond and upper_cond

    # ================= #
    # Private methods.
    # ================= #

    def _add_mjcf_elements(self) -> None:
        # Parse joints.
        self._joints = []
        self._joint_elem_mapping = {}
        for joint in consts.Joints:
            joint_elem = self._mjcf_root.find("joint", joint.name)
            self._joints.append(joint_elem)
            self._joint_elem_mapping[joint] = joint_elem

        # Parse fingertip sites.
        self._fingertip_sites: List[mjcf.Element] = []
        for tip_name in consts.FINGERTIP_NAMES:
            tip_elem = self._mjcf_root.find("body", tip_name)
            tip_site = tip_elem.add(
                "site",
                name=tip_name + "_site",
                pos="0 0 0",
                # NOTE(kevin): The kwargs below are for visualization purposes.
                size="0.001 0.001 0.001",
                type="sphere",
                rgba="1 0 0 1",
            )
            self._fingertip_sites.append(tip_site)

    def _add_tendons(self) -> None:
        """Add tendons to the hand."""
        self._tendons = []
        self._tendon_elem_mapping = {}
        for tendon, joints in consts.TENDON_JOINT_MAPPING.items():
            tendon_elem = self._mjcf_root.tendon.add("fixed", name=tendon.name)
            for joint in joints:
                # We set `coef=1` to make the tendon's length equal to the sum of the
                # joint positions.
                tendon_elem.add("joint", joint=joint.name, coef=1.0)
            self._tendons.append(tendon_elem)
            self._tendon_elem_mapping[tendon] = tendon_elem

    def _add_actuators(self) -> None:
        """Adds actuators to the hand."""
        if self._actuation not in consts.Actuation:
            raise ValueError(f"Actuation {self._actuation} is not a valid actuation.")

        if self._actuation == consts.Actuation.POSITION:
            self._add_position_actuators()

    def _add_position_actuators(self) -> None:
        """Adds position actuators to the mjcf model."""

        def add_actuator(act: consts.Actuators, params: _ActuatorParams) -> MjcfElement:
            # Create a position actuator mjcf elem.
            actuator = self._mjcf_root.actuator.add(
                "position",
                name=act.name,
                ctrllimited=True,
                ctrlrange=consts.ACTUATION_LIMITS[self._actuation][act],
                forcelimited=True,
                forcerange=consts.EFFORT_LIMITS[act],
                kp=params.kp,
            )

            # NOTE(kevin): When specifying the joint or tendon to which an actuator is
            # attached, the mjcf element itself is used, rather than the name string.
            if act in consts.ACTUATOR_TENDON_MAPPING:
                actuator.tendon = self._tendon_elem_mapping[
                    consts.ACTUATOR_TENDON_MAPPING[act]
                ]
            else:
                actuator.joint = self._joint_elem_mapping[
                    consts.ACTUATOR_JOINT_MAPPING[act][0]
                ]

            return actuator

        self._actuators = []
        for actuator, actuator_params in _ACTUATOR_PARAMS[self._actuation].items():
            self._actuators.append(add_actuator(actuator, actuator_params))

    def _add_sensors(self) -> None:
        """Add sensors to the mjcf model."""
        self._add_torque_sensors()

    def _add_torque_sensors(self) -> None:
        """Adds torque sensors to the joints of the hand."""
        self._joint_torque_sensors = []
        self._joint_torque_sensor_elem_mapping = {}
        for joint in consts.Joints:
            joint_elem = self._joint_elem_mapping[joint]
            site_elem = joint_elem.parent.add(
                "site",
                name=joint_elem.name + "_site",
                # NOTE(kevin): The kwargs below are for visualization purposes.
                size="0.001 0.001 0.001",
                type="box",
                rgba="0 1 0 1",
            )
            # Create a 3-axis torque sensor.
            torque_sensor_elem = joint_elem.root.sensor.add(
                "torque",
                site=site_elem,
                name=joint_elem.name + "_torque",
            )
            self._joint_torque_sensors.append(torque_sensor_elem)
            self._joint_torque_sensor_elem_mapping[joint] = torque_sensor_elem

    def _color_hand(self) -> None:
        """Assigns a random RGB color to the hand."""
        for geom_name in consts.COLORED_GEOMS:
            geom = self._mjcf_root.find("geom", geom_name)
            rgb = np.random.uniform(size=3).flatten()
            rgba = np.append(rgb, 1)
            geom.rgba = rgba
