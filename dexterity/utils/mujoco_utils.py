from typing import Sequence

import mujoco
import numpy as np
from dm_control import mjcf

from dexterity import hints


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
    idx = physics.model.name2id(site_elem.full_identifier, mujoco.mjtObj.mjOBJ_SITE)
    site_vel = np.empty(6)
    mujoco.mj_objectVelocity(
        physics.model.ptr,
        physics.data.ptr,
        mujoco.mjtObj.mjOBJ_SITE,
        idx,
        site_vel,
        flg_local,
    )
    return np.hstack([site_vel[3:], site_vel[:3]])


def compute_object_6d_jacobian(
    model: hints.MjModel,
    data: hints.MjData,
    object_type: hints.MujocoObjectType,
    object_id: int,
) -> np.ndarray:
    """Computes the (6, nv) object Jacobian.

    The Jacobian maps joint velocities to the object's Cartesian 6D velocity in the
    world frame. Each row of the Jacobian is the gradient of the corresponding 3D
    coordinate of the specified point with respect to the degrees of freedom.

    Only MuJoCo bodies, geoms and sites are supported.
    """
    jacobian = np.empty((6, model.nv), dtype=data.qpos.dtype)
    jacobian_position, jacobian_rotation = jacobian[:3], jacobian[3:]

    if object_type == mujoco.mjtObj.mjOBJ_BODY:
        func = mujoco.mj_jacBody
    elif object_type == mujoco.mjtObj.mjOBJ_GEOM:
        func = mujoco.mj_jacGeom
    elif object_type == mujoco.mjtObj.mjOBJ_SITE:
        func = mujoco.mj_jacSite
    else:
        raise ValueError(
            f"Invalid `object_type` {object_type}. Only bodies, geoms and sites"
            " are supported."
        )

    func(
        model.ptr,
        data.ptr,
        jacobian_position,
        jacobian_rotation,
        object_id,
    )

    return jacobian


def get_element_type(element: hints.MjcfElement) -> mujoco.mjtObj:
    if element.tag == "body":
        return mujoco.mjtObj.mjOBJ_BODY
    elif element.tag == "geom":
        return mujoco.mjtObj.mjOBJ_GEOM
    elif element.tag == "site":
        return mujoco.mjtObj.mjOBJ_SITE
    else:
        raise ValueError(
            f"Element must be a MuJoCo body, geom or site. Got [{element.tag}]."
        )


def compensate_gravity(
    physics: mjcf.Physics, body_elements: Sequence[hints.MjcfElement]
) -> None:
    """Counteracts gravity by applying forces to body elements."""
    gravity = np.hstack([physics.model.opt.gravity, [0.0, 0.0, 0.0]])
    physics_bodies = physics.bind(body_elements)
    if physics_bodies is None:
        raise ValueError("Calling bind() on the body elements returned None.")
    physics_bodies.xfrc_applied[:] = -gravity * physics_bodies.mass[..., None]
