from dm_control import mjcf

from shadow_hand import shadow_hand_e_constants as consts


class ShadowHandSeriesE:
    """Shadow Dexterous Hand E Series."""

    def __init__(self, name: str = "shadow_hand_e") -> None:
        self.mjcf_model = mjcf.RootElement(model=name)

        # self._set_options()
        # self._set_defaults()

        self._include_shared_options()
        self._include_hand_options()
        self._include_shared_assets()

        self._add_forearm()
        self._add_wrist()
        self._add_palm()

        self._add_first_finger()
        self._add_middle_finger()
        self._add_ring_finger()
        self._add_little_finger()
        self._add_thumb()

    def _set_options(self) -> None:
        # Compiler options.
        self.mjcf_model.compiler.angle = "radian"
        self.mjcf_model.compiler.coordinate = "local"
        # self.mjcf_model.compiler.meshdir = ""
        # self.mjcf_model.compiler.texturedir = ""

        # # Visual options.
        # self.mjcf_model.visual.map.fogstart = 3
        # self.mjcf_model.visual.map.fogend = 5
        # self.mjcf_model.visual.map.force = 0.1
        # self.mjcf_model.visual.map.znear = 0.1
        # self.mjcf_model.visual.quality.shadowsize = 2048
        # self.mjcf_model.visual.quality.offsamples = 8
        # setattr(getattr(self.mjcf_model.visual, "global"), "offwidth", 800)
        # setattr(getattr(self.mjcf_model.visual, "global"), "offheight", 800)

        # self.mjcf_model.option.cone = "elliptic"
        # self.mjcf_model.option.impratio = 300
        # self.mjcf_model.option.timestep = 0.02

        # self.mjcf_model.size.njmax = 500
        # self.mjcf_model.size.nconmax = 200

    def _set_defaults(self) -> None:
        self.mjcf_model.default.mesh.scale = (0.001, 0.001, 0.001)

        self.mjcf_model.default.joint.limited = True
        self.mjcf_model.default.joint.damping = 0.05
        self.mjcf_model.default.joint.armature = 0.005
        self.mjcf_model.default.joint.margin = 0.0
        self.mjcf_model.default.joint.frictionloss = 0.1
        # from ipdb import set_trace; set_trace()

    def _include_shared_options(self) -> None:
        shared_options_path = str(consts._SRC_ROOT / "shared_options.xml")
        shared_options_mjcf = mjcf.from_path(shared_options_path)
        self.mjcf_model.include_copy(shared_options_mjcf)

    def _include_shared_assets(self) -> None:
        shared_assets_path = str(consts._SRC_ROOT / "shared_assets.xml")
        shared_assets_mjcf = mjcf.from_path(shared_assets_path)
        self.mjcf_model.include_copy(shared_assets_mjcf)

    def _include_hand_options(self) -> None:
        hand_e_options_path = str(consts._SRC_ROOT / "sr_hand_e_options.xml")
        hand_e_options_mjcf = mjcf.from_path(hand_e_options_path)
        self.mjcf_model.include_copy(hand_e_options_mjcf)

    def _add_visual(
        self,
        element: mjcf.Element,
        **kwargs,
    ) -> None:
        element.add(
            "geom",
            **kwargs,
            contype="0",
            conaffinity="0",
            group="0",
        )

    def _add_collision(
        self,
        element: mjcf.Element,
        **kwargs,
    ) -> None:
        element.add(
            "geom",
            **kwargs,
            group="3",
        )

    def _add_forearm(self) -> None:
        self.forearm = self.mjcf_model.worldbody.add(
            "body",
            name="forearm",
        )
        self.forearm.add(
            "inertial",
            pos="0 0 0.09",
            mass="3",
            diaginertia="0.0138 0.0138 0.00744",
        )
        # Visual.
        self._add_visual(
            self.forearm,
            type="mesh",
            material="plastic",
            mesh="forearm_visual",
        )
        # Collision.
        self._add_collision(
            self.forearm,
            type="mesh",
            material="plastic",
            mesh="forearm_collision",
        )
        self._add_collision(
            self.forearm,
            type="mesh",
            material="plastic",
            mesh="forearm_collision_wrist",
        )

    def _add_wrist(self) -> None:
        self.wrist = self.forearm.add(
            "body",
            name="wrist",
            pos="0 -0.01 0.213",
        )
        self.wrist.add(
            "inertial",
            pos="0 0 0.029",
            quat="0.5 0.5 0.5 0.5",
            mass="0.1",
            diaginertia="6.4e-05 4.38e-05 3.5e-05",
        )
        self.wrist.add(
            "joint",
            name="WRJ2",
            axis="0 1 0",
            range="-0.523599 0.174533",
            damping="1.0",
            frictionloss="0.1",
        )
        # Visual.
        self._add_visual(
            self.wrist,
            type="mesh",
            material="metal",
            mesh="wrist_visual",
        )
        # Collision.
        for i in range(1, 8):
            self._add_collision(
                self.wrist,
                type="mesh",
                material="metal",
                mesh="wrist_collision_{}".format(i),
            )

    def _add_palm(self) -> None:
        self.palm = self.wrist.add(
            "body",
            name="palm",
            pos="0 0 0.034",
        )
        self.palm.add(
            "inertial",
            pos="0 0 0.035",
            quat="0.707107 0 0 0.707107",
            mass="0.3",
            diaginertia="0.0005287 0.0003581 0.000191",
        )
        self.palm.add(
            "joint",
            name="WRJ1",
            axis="1 0 0",
            range="-0.698132 0.488692",
            damping="1.0",
            frictionloss="0.1",
        )
        # Visual.
        self.palm.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="palm_visual",
            dclass="visual",
        )
        # Collision.
        for i in range(145):
            self.palm.add(
                "geom",
                type="mesh",
                material="plastic",
                mesh="palm_collision_{}".format(i),
                dclass="collision",
            )
        # Add .
        self.palm_manipulator = self.palm.add(
            "body",
            name="manipulator",
            pos="0 0 0.05",
        )
        self.palm_manipulator.add(
            "inertial",
            pos="0 0 0.05",
            mass="0",
            diaginertia="0 0 0",
        )
        # Add IMU.
        self.palm_imu = self.palm.add(
            "body",
            name="imu",
            pos="0.01785 0.00765 0.049125",
            quat="2.31079e-07 -2.31079e-07 0.707107 0.707107",
        )
        self.palm_imu.add(
            "inertial",
            pos="0.01785 0.00765 0.049125",
            quat="2.31079e-07 -2.31079e-07 0.707107 0.707107",
            mass="0",
            diaginertia="0 0 0",
        )

    def _add_first_finger(self) -> None:
        # FF knuckle.
        self.first_finger_knuckle = self.palm.add(
            "body",
            name="ffknuckle",
            pos="0.033 0 0.095",
        )
        self.first_finger_knuckle.add(
            "inertial",
            pos="0 0 0",
            quat="0.5 0.5 -0.5 0.5",
            mass="0.008",
            diaginertia="3.2e-07 2.6e-07 2.6e-07",
        )
        self.first_finger_knuckle.add(
            "joint",
            name="FFJ4",
            dclass="FJ4",
            axis="0 -1 0",
        )
        # Visual.
        self.first_finger_knuckle.add(
            "geom",
            type="mesh",
            material="metal",
            mesh="knuckle_visual",
            dclass="visual",
        )
        # Collision.
        self.first_finger_knuckle.add(
            "geom",
            type="mesh",
            material="metal",
            mesh="knuckle_collision",
            dclass="collision",
        )

        # FF proximal.
        self.first_finger_proximal = self.first_finger_knuckle.add(
            "body",
            name="ffproximal",
        )
        self.first_finger_proximal.add(
            "joint",
            name="FFJ3",
            dclass="FJ3",
        )
        self.first_finger_proximal.add(
            "inertial",
            pos="0 0 0.0225",
            quat="0.707107 0 0 0.707107",
            mass="0.03",
            diaginertia="1e-05 9.8e-06 1.8e-06",
        )
        # Visual.
        self.first_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_visual",
            dclass="visual",
        )
        # Collision.
        self.first_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_collision_proximal",
            dclass="collision",
        )
        self.first_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_collision_distal",
            dclass="collision",
        )

        # FF middle.
        self.first_finger_middle = self.first_finger_proximal.add(
            "body",
            name="ffmiddle",
            pos="0 0 0.045",
        )
        self.first_finger_middle.add(
            "inertial",
            pos="0 0 0.0125",
            quat="0.707107 0 0 0.707107",
            mass="0.017",
            diaginertia="2.7e-06 2.6e-06 8.7e-07",
        )
        self.first_finger_middle.add(
            "joint",
            name="FFJ2",
            dclass="FJ2",
        )
        self.first_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_visual",
            dclass="visual",
        )
        self.first_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_proximal",
            dclass="collision",
        )
        self.first_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_middle",
            dclass="collision",
        )
        self.first_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_distal",
            dclass="collision",
        )

        # FF distal.
        self.first_finger_distal = self.first_finger_middle.add(
            "body",
            name="ffdistal",
            pos="0 0 0.025",
        )
        self.first_finger_distal.add(
            "inertial",
            pos="0 0 0.012",
            quat="0.707107 0 0 0.707107",
            mass="0.012",
            diaginertia="1.1e-06 9.4e-07 5.3e-07",
        )
        self.first_finger_distal.add(
            "joint",
            name="FFJ1",
            dclass="FJ1",
        )
        self.first_finger_distal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F1_visual",
            dclass="visual",
        )
        self.first_finger_distal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F1_collision",
            dclass="collision",
        )

        # FF tip.
        self.first_finger_tip = self.first_finger_distal.add(
            "body", name="fftip", pos="0 0 0.026"
        )
        self.first_finger_tip.add(
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="0 0 0",
        )

    def _add_middle_finger(self) -> None:
        # MF knuckle.
        self.middle_finger_knuckle = self.palm.add(
            "body",
            name="mfknuckle",
            pos="0.011 0 0.099",
        )
        self.middle_finger_knuckle.add(
            "inertial",
            pos="0 0 0",
            quat="0.5 0.5 -0.5 0.5",
            mass="0.008",
            diaginertia="3.2e-07 2.6e-07 2.6e-07",
        )
        self.middle_finger_knuckle.add(
            "joint",
            name="MFJ4",
            dclass="FJ4",
            axis="0 -1 0",
        )
        # Visual.
        self.middle_finger_knuckle.add(
            "geom",
            type="mesh",
            material="metal",
            mesh="knuckle_visual",
            dclass="visual",
        )
        # Collision.
        self.middle_finger_knuckle.add(
            "geom",
            type="mesh",
            material="metal",
            mesh="knuckle_collision",
            dclass="collision",
        )

        # MF proximal.
        self.middle_finger_proximal = self.middle_finger_knuckle.add(
            "body",
            name="mfproximal",
        )
        self.middle_finger_proximal.add(
            "joint",
            name="MFJ3",
            dclass="FJ3",
        )
        self.middle_finger_proximal.add(
            "inertial",
            pos="0 0 0.0225",
            quat="0.707107 0 0 0.707107",
            mass="0.03",
            diaginertia="1e-05 9.8e-06 1.8e-06",
        )
        # Visual.
        self.middle_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_visual",
            dclass="visual",
        )
        # Collision.
        self.middle_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_collision_proximal",
            dclass="collision",
        )
        self.middle_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_collision_distal",
            dclass="collision",
        )

        # MF middle.
        self.middle_finger_middle = self.middle_finger_proximal.add(
            "body",
            name="mfmiddle",
            pos="0 0 0.045",
        )
        self.middle_finger_middle.add(
            "inertial",
            pos="0 0 0.0125",
            quat="0.707107 0 0 0.707107",
            mass="0.017",
            diaginertia="2.7e-06 2.6e-06 8.7e-07",
        )
        self.middle_finger_middle.add(
            "joint",
            name="MFJ2",
            dclass="FJ2",
        )
        # Visual.
        self.middle_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_visual",
            dclass="visual",
        )
        # Collision.
        self.middle_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_proximal",
            dclass="collision",
        )
        self.middle_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_middle",
            dclass="collision",
        )
        self.middle_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_distal",
            dclass="collision",
        )

        # MF distal.
        self.middle_finger_distal = self.middle_finger_middle.add(
            "body",
            name="mfdistal",
            pos="0 0 0.025",
        )
        self.middle_finger_distal.add(
            "inertial",
            pos="0 0 0.012",
            quat="0.707107 0 0 0.707107",
            mass="0.012",
            diaginertia="1.1e-06 9.4e-07 5.3e-07",
        )
        self.middle_finger_distal.add(
            "joint",
            name="MFJ1",
            dclass="FJ1",
        )
        # Visual.
        self.middle_finger_distal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F1_visual",
            dclass="visual",
        )
        # Collision.
        self.middle_finger_distal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F1_collision",
            dclass="collision",
        )

        # MF tip.
        self.middle_finger_tip = self.middle_finger_distal.add(
            "body",
            name="mftip",
            pos="0 0 0.026",
        )
        self.middle_finger_tip.add(
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="0 0 0",
        )

    def _add_ring_finger(self) -> None:
        # RF knuckle.
        self.ring_finger_knuckle = self.palm.add(
            "body",
            name="rfknuckle",
            pos="-0.011 0 0.095",
        )
        self.ring_finger_knuckle.add(
            "inertial",
            pos="0 0 0",
            quat="0.5 0.5 -0.5 0.5",
            mass="0.008",
            diaginertia="3.2e-07 2.6e-07 2.6e-07",
        )
        self.ring_finger_knuckle.add(
            "joint",
            name="RFJ4",
            dclass="FJ4",
            axis="0 1 0",
        )
        # Visual.
        self.ring_finger_knuckle.add(
            "geom",
            type="mesh",
            material="metal",
            mesh="knuckle_visual",
            dclass="visual",
        )
        # Collision.
        self.ring_finger_knuckle.add(
            "geom",
            type="mesh",
            material="metal",
            mesh="knuckle_collision",
            dclass="collision",
        )

        # RF proximal.
        self.ring_finger_proximal = self.ring_finger_knuckle.add(
            "body",
            name="rfproximal",
        )
        self.ring_finger_proximal.add(
            "inertial",
            pos="0 0 0.0225",
            quat="0.707107 0 0 0.707107",
            mass="0.03",
            diaginertia="1e-05 9.8e-06 1.8e-06",
        )
        self.ring_finger_proximal.add(
            "joint",
            name="RFJ3",
            dclass="FJ3",
        )
        # Visual.
        self.ring_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_visual",
            dclass="visual",
        )
        # Collision.
        self.ring_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_collision_proximal",
            dclass="collision",
        )
        self.ring_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_collision_distal",
            dclass="collision",
        )

        # RF middle.
        self.ring_finger_middle = self.ring_finger_proximal.add(
            "body",
            name="rfmiddle",
            pos="0 0 0.045",
        )
        self.ring_finger_middle.add(
            "inertial",
            pos="0 0 0.0125",
            quat="0.707107 0 0 0.707107",
            mass="0.017",
            diaginertia="2.7e-06 2.6e-06 8.7e-07",
        )
        self.ring_finger_middle.add(
            "joint",
            name="RFJ2",
            dclass="FJ2",
        )
        # Visual.
        self.ring_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_visual",
            dclass="visual",
        )
        # Collision.
        self.ring_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_proximal",
            dclass="collision",
        )
        self.ring_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_middle",
            dclass="collision",
        )
        self.ring_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_distal",
            dclass="collision",
        )

        # RF distal.
        self.ring_finger_distal = self.ring_finger_middle.add(
            "body",
            name="rfdistal",
            pos="0 0 0.025",
        )
        self.ring_finger_distal.add(
            "inertial",
            pos="0 0 0.012",
            quat="0.707107 0 0 0.707107",
            mass="0.012",
            diaginertia="1.1e-06 9.4e-07 5.3e-07",
        )
        self.ring_finger_distal.add(
            "joint",
            name="RFJ1",
            dclass="FJ1",
        )
        # Visual.
        self.ring_finger_distal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F1_visual",
            dclass="visual",
        )
        # Collision.
        self.ring_finger_distal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F1_collision",
            dclass="collision",
        )

        # RF tip.
        self.ring_finger_tip = self.ring_finger_distal.add(
            "body",
            name="rftip",
            pos="0 0 0.026",
        )
        self.ring_finger_tip.add(
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="0 0 0",
        )

    def _add_little_finger(self) -> None:
        # LF metacarpal.
        self.little_finger_metacarpal = self.palm.add(
            "body",
            name="lfmetacarpal",
            pos="-0.033 0 0.02071",
        )
        self.little_finger_metacarpal.add(
            "inertial",
            pos="0 0 0.04",
            quat="0.707107 0 0 0.707107",
            mass="0.03",
            diaginertia="1.638e-05 1.45e-05 4.272e-06",
        )
        self.little_finger_metacarpal.add(
            "joint",
            name="LFJ5",
            axis="0.573576 0 0.819152",
            range="0 0.785398",
        )
        # Visual.
        self.little_finger_metacarpal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="lfmetacarpal_visual",
            dclass="visual",
        )
        # Collision.
        for i in range(16):
            self.little_finger_metacarpal.add(
                "geom",
                type="mesh",
                material="plastic",
                mesh=f"lfmetacarpal_collision_{i}",
                dclass="collision",
            )

        # LF knuckle.
        self.little_finger_knuckle = self.little_finger_metacarpal.add(
            "body",
            name="lfknuckle",
            pos="0 0 0.06579",
        )
        self.little_finger_knuckle.add(
            "inertial",
            pos="0 0 0",
            quat="0.5 0.5 -0.5 0.5",
            mass="0.008",
            diaginertia="3.2e-07 2.6e-07 2.6e-07",
        )
        self.little_finger_knuckle.add(
            "joint",
            name="LFJ4",
            dclass="FJ4",
            axis="0 1 0",
        )
        # Visual.
        self.little_finger_knuckle.add(
            "geom",
            type="mesh",
            material="metal",
            mesh="knuckle_visual",
            dclass="visual",
        )
        # Collision.
        self.little_finger_knuckle.add(
            "geom",
            type="mesh",
            material="metal",
            mesh="knuckle_collision",
            dclass="collision",
        )

        # LF proximal.
        self.little_finger_proximal = self.little_finger_knuckle.add(
            "body",
            name="lfproximal",
        )
        self.little_finger_proximal.add(
            "inertial",
            pos="0 0 0.0225",
            quat="0.707107 0 0 0.707107",
            mass="0.03",
            diaginertia="1e-05 9.8e-06 1.8e-06",
        )
        self.little_finger_proximal.add(
            "joint",
            name="LFJ3",
            dclass="FJ3",
        )
        # Visual.
        self.little_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_visual",
            dclass="visual",
        )
        # Collision.
        self.little_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_collision_proximal",
            dclass="collision",
        )
        self.little_finger_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F3_collision_distal",
            dclass="collision",
        )

        # LF middle.
        self.little_finger_middle = self.little_finger_proximal.add(
            "body",
            name="lfmiddle",
            pos="0 0 0.045",
        )
        self.little_finger_middle.add(
            "inertial",
            pos="0 0 0.0125",
            quat="0.707107 0 0 0.707107",
            mass="0.017",
            diaginertia="2.7e-06 2.6e-06 8.7e-07",
        )
        self.little_finger_middle.add(
            "joint",
            name="LFJ2",
            dclass="FJ2",
        )
        # Visual.
        self.little_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_visual",
            dclass="visual",
        )
        # Collision.
        self.little_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_proximal",
            dclass="collision",
        )
        self.little_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_middle",
            dclass="collision",
        )
        self.little_finger_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F2_collision_distal",
            dclass="collision",
        )

        # LF distal.
        self.little_finger_distal = self.little_finger_middle.add(
            "body",
            name="lfdistal",
            pos="0 0 0.025",
        )
        self.little_finger_distal.add(
            "inertial",
            pos="0 0 0.012",
            quat="0.707107 0 0 0.707107",
            mass="0.012",
            diaginertia="1.1e-06 9.4e-07 5.3e-07",
        )
        self.little_finger_distal.add(
            "joint",
            name="LFJ1",
            dclass="FJ1",
        )
        # Visual.
        self.little_finger_distal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F1_visual",
            dclass="visual",
        )
        # Collision.
        self.little_finger_distal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="F1_collision",
            dclass="collision",
        )

        # LF tip.
        self.little_finger_tip = self.little_finger_distal.add(
            "body",
            name="lftip",
            pos="0 0 0.026",
        )
        self.little_finger_tip.add(
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="0 0 0",
        )

    def _add_thumb(self) -> None:
        # Thumb base.
        self.thumb_base = self.palm.add(
            "body",
            name="thbase",
            pos="0.034 -0.0085 0.029",
            quat="0.92388 0 0.382683 0",
        )
        self.thumb_base.add(
            "inertial",
            pos="0 0 0",
            mass="0.01",
            diaginertia="1.6e-07 1.6e-07 1.6e-07",
        )
        self.thumb_base.add(
            "joint",
            name="THJ5",
            axis="0 0 -1",
            range="-1.0472 1.0472",
            dclass="THJ5",
        )

        # Thumb proximal.
        self.thumb_proximal = self.thumb_base.add(
            "body",
            name="thproximal",
        )
        self.thumb_proximal.add(
            "inertial",
            pos="0 0 0.019",
            mass="0.04",
            diaginertia="1.36e-05 1.36e-05 3.13e-06",
        )
        self.thumb_proximal.add(
            "joint",
            name="THJ4",
            axis="1 0 0",
            range="0 1.22173",
            dclass="THJ4",
        )
        # Visual.
        self.thumb_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="TH3_visual",
            dclass="visual",
        )
        # Collision.
        self.thumb_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="TH3_collision_proximal",
            dclass="collision",
        )
        self.thumb_proximal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="TH3_collision_distal",
            dclass="collision",
        )

        # Thumb hub.
        self.thumb_hub = self.thumb_proximal.add(
            "body",
            name="thhub",
            pos="0 0 0.038",
        )
        self.thumb_hub.add(
            "inertial",
            pos="0 0 0",
            mass="0.005",
            diaginertia="1.0e-06 1.0e-06 3.0e-07",
        )
        self.thumb_hub.add(
            "joint",
            name="THJ3",
            axis="1 0 0",
            range="-0.20944 0.20944",
            dclass="THJ3",
        )

        # Thumb middle.
        self.thumb_middle = self.thumb_hub.add(
            "body",
            name="thmiddle",
        )
        self.thumb_middle.add(
            "inertial",
            pos="0 0 0.016",
            mass="0.02",
            diaginertia="5.1e-06 5.1e-06 1.21e-06",
        )
        self.thumb_middle.add(
            "joint",
            name="THJ2",
            axis="0 -1 0",
            range="-0.698132 0.698132",
            dclass="THJ2",
        )
        # Visual.
        self.thumb_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="TH2_visual",
            dclass="visual",
        )
        # Collision.
        self.thumb_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="TH2_collision_proximal",
            dclass="collision",
        )
        self.thumb_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="TH2_collision_middle",
            dclass="collision",
        )
        self.thumb_middle.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="TH2_collision_distal",
            dclass="collision",
        )

        # Thumb distal.
        self.thumb_distal = self.thumb_middle.add(
            "body",
            name="thdistal",
            pos="0 0 0.032",
            quat="0.707107 0 0 -0.707107",
        )
        self.thumb_distal.add(
            "inertial",
            pos="0 0 0.01375",
            quat="0.707107 0 0 0.707107",
            mass="0.016",
            diaginertia="2.2e-06 2.1e-06 1e-06",
        )
        self.thumb_distal.add(
            "joint",
            name="THJ1",
            axis="1 0 0",
            range="0 1.5708",
            dclass="THJ1",
        )
        # Visual.
        self.thumb_distal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="TH1_visual",
            dclass="visual",
        )
        # Collision.
        self.thumb_distal.add(
            "geom",
            type="mesh",
            material="plastic",
            mesh="TH1_collision",
            dclass="collision",
        )

        # Thumb tip.
        self.thumb_tip = self.thumb_distal.add(
            "body",
            name="thtip",
            pos="0 0 0.0275",
        )
        self.thumb_tip.add(
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="0 0 0",
        )


if __name__ == "__main__":
    hand = ShadowHandSeriesE()
    physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
    physics.step()
