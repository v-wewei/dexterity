import enum

import numpy as np
from dm_control import mjcf

from shadow_hand.hints import MjcfElement

# from dm_robotics.transformations import transformations as tr


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


def hmat_inv(hmat: np.ndarray) -> np.ndarray:
    """Numerically stable inverse of homogeneous transform."""
    rot = hmat[0:3, 0:3]
    pos = hmat[0:3, 3]
    hinv = np.eye(4)
    hinv[0:3, 3] = rot.T.dot(-pos)
    hinv[0:3, 0:3] = rot.T
    return hinv


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
    pose_bw = hmat_inv(pose_wb)  # Pose of world frame in site_b.
    pose_ba = pose_bw @ pose_wa  # Pose of site_a in site_b.
    return pose_ba


def get_site_velocity(
    physics: mjcf.Physics, site_elem: mjcf.Element, world_frame: bool = False
) -> np.ndarray:
    raise NotImplementedError


def get_site_relative_velocity(
    physics: mjcf.Physics,
    site_a: MjcfElement,
    site_b: MjcfElement,
    frame: Frame = Frame.WORLD,
) -> np.ndarray:
    raise NotImplementedError
