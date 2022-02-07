import imageio
import numpy as np
from dm_control import mjcf, mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_robotics.transformations import transformations as tr

from shadow_hand.models.arenas.empty import Arena
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts

enums = mjbindings.enums
mjlib = mjbindings.mjlib


def render(physics: mjcf.Physics, cam_id: str = "fixed_viewer") -> np.ndarray:
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.frame = enums.mjtFrame.mjFRAME_GEOM
    return physics.render(
        width=640, height=480, camera_id=cam_id, scene_option=scene_option
    )


def main() -> None:
    # Build the arena.
    arena = Arena("hand_arena")

    # Load the hand and add it to the arena.
    attachment_site = arena.mjcf_model.worldbody.add(
        "site",
        type="sphere",
        pos=[0, 0, 0.08],
        rgba="0 0 0 0",
        size="0.01",
    )
    hand = shadow_hand_e.ShadowHandSeriesE(actuation=consts.Actuation.POSITION)
    arena.attach(hand, attachment_site)

    # TODO(kevin): Figure out how to draw axis of rotation.
    # mjtGeom->mjGEOM_ARROW (rendering-only geom type)

    # Compile.
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    frames = []
    axis_angle = np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
    for mag in np.linspace(0, np.pi):
        hand.set_pose(physics, quaternion=tr.axisangle_to_quat(mag * axis_angle))
        physics.step()
        frames.append(render(physics))
    imageio.mimsave("temp/scene.gif", frames, fps=30)


if __name__ == "__main__":
    main()
