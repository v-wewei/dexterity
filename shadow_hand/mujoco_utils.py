import enum

import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_robotics.transformations import transformations as tr

from shadow_hand.hints import MjcfElement

mjlib = mjbindings.mjlib


class Frame(enum.Enum):
    WORLD = enum.auto()
    A = enum.auto()
    B = enum.auto()


def get_site_pose(physics: mjcf.Physics, site_elem: MjcfElement) -> np.ndarray:
    """Returns the world pose of the site as a 4x4 transform.

    Args:
        physics: An `mjcf.Physics` instance.
        site_entity: An `mjcf.Element` instance.
    """
    binding = physics.bind(site_elem)
    xpos = binding.xpos.reshape(3, 1)
    xmat = binding.xmat.reshape(3, 3)
    return np.vstack(
        [
            np.hstack([xmat, xpos]),
            np.array([0, 0, 0, 1]),
        ]
    )


def get_site_relative_pose(
    physics: mjcf.Physics, site_a: MjcfElement, site_b: MjcfElement
) -> np.ndarray:
    """Returns the pose of `site_a` in the frame of `site_b`.

    Args:
        physics: An `mjcf.Physics` instance.
        site_a: An `mjcf.Element` instance.
        site_b: An `mjcf.Element` instance.
    """
    pose_wa = get_site_pose(physics, site_a)  # Pose of site_a in world frame.
    pose_wb = get_site_pose(physics, site_b)  # Pose of site_b in world frame.
    pose_bw = tr.hmat_inv(pose_wb)  # Pose of world frame in site_b.
    pose_ba = pose_bw @ pose_wa  # Pose of site_a in site_b.
    return pose_ba


def get_site_velocity(
    physics: mjcf.Physics, site_elem: mjcf.Element, world_frame: bool = False
) -> np.ndarray:
    """Returns the linear and angular velocities of the site.

    This 6-D vector represents the instantaneous velocity of the coordinate system
    attached to the site. If `world_frame=True`, the velocity is expressed in the world
    frame.

    Args:
        physics: An `mjcf.Physics` instance.
        site_elem: An `mjcf.Element` instance.
        world_frame: Whether to return the velocity in the world frame.
    """
    flg_local = 0 if world_frame else 1
    idx = physics.model.name2id(site_elem.full_identifier, enums.mjtObj.mjOBJ_SITE)
    site_vel = np.empty(6)
    mjlib.mj_objectVelocity(
        physics.model.ptr,
        physics.data.ptr,
        enums.mjtObj.mjOBJ_SITE,
        idx,
        site_vel,
        flg_local,
    )
    return np.hstack([site_vel[3:], site_vel[:3]])


def get_site_relative_velocity(
    physics: mjcf.Physics,
    site_a: MjcfElement,
    site_b: MjcfElement,
    frame: Frame = Frame.WORLD,
) -> np.ndarray:
    raise NotImplementedError
