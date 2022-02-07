from typing import List

# import imageio
import numpy as np
from dm_control import mjcf, mujoco

# from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.utils import inverse_kinematics
from dm_robotics.transformations import transformations as tr
from matplotlib import pyplot as plt

from shadow_hand.models.arenas.empty import Arena
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts


def render(physics: mjcf.Physics, cam_id: str = "fixed_viewer1") -> np.ndarray:
    scene_option = mujoco.wrapper.core.MjvOption()
    # scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
    return physics.render(
        width=640, height=480, camera_id=cam_id, scene_option=scene_option
    )


def animate(
    physics: mjcf.Physics,
    frames: List[np.ndarray] = [],
    duration: float = 2.0,
    framerate: float = 30,
) -> None:
    scene_option = mujoco.wrapper.core.MjvOption()
    # scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
    while physics.data.time < duration:
        physics.step()
        if len(frames) < physics.data.time * framerate:
            pixels = physics.render(
                width=640,
                height=480,
                camera_id="fixed_viewer1",
                scene_option=scene_option,
            )
            frames.append(pixels)


def main() -> None:
    # Build the arena.
    arena = Arena("hand_arena")
    arena.mjcf_model.option.gravity = (0.0, 0.0, 0.0)

    target_pos = np.array([0.01099978, -0.40800001, 0.17000004])
    arena.mjcf_model.worldbody.add(
        "site",
        type="sphere",
        pos=target_pos,
        rgba="0 0 1 1",
        size="0.005",
    )

    # Load the hand and add it to the arena.
    attachment_site = arena.mjcf_model.worldbody.add(
        "site",
        type="sphere",
        pos=[0, 0, 0.1],
        rgba="0 0 0 0",
        size="0.01",
    )
    hand = shadow_hand_e.ShadowHandSeriesE(actuation=consts.Actuation.POSITION)
    arena.attach(hand, attachment_site)

    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    axis_angle = np.pi * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
    hand.set_pose(physics, quaternion=tr.axisangle_to_quat(axis_angle))
    physics.step()

    finger = consts.Components.RF
    fingertip_name = finger.name.lower() + "tip"
    site_name = fingertip_name + "_site"
    site_name = f"{hand.mjcf_model.model}/{site_name}"
    wrist_joints = consts.JOINT_GROUP[consts.Components.WR]
    finger_joints = consts.JOINT_GROUP[finger]
    joints_group = wrist_joints + finger_joints

    # Get elems for each joints.
    joints_group_elems = []
    for joint, joint_elem in hand._joint_elem_mapping.items():
        if joint in joints_group:
            joints_group_elems.append(joint_elem)
    assert len(joints_group) == len(joints_group_elems)
    joint_names = [joint.full_identifier for joint in joints_group_elems]

    max_ik_attempts = 10
    for i in range(max_ik_attempts):
        result = inverse_kinematics.qpos_from_site_pose(
            physics=physics,
            site_name=site_name,
            target_pos=target_pos,
            joint_names=joint_names,
            inplace=False,
            tol=1e-14,
            regularization_strength=1e-2,
        )
        success = result.success

        print(f"({i}) success: {success}")

        if success or max_ik_attempts <= 1:
            break
        else:
            # Randomize joints.
            lower = [jl[0] for jl in consts.JOINT_LIMITS.values()]
            upper = [jl[1] for jl in consts.JOINT_LIMITS.values()]
            physics.bind(hand.joints).qpos = np.random.uniform(lower, upper)

    # print("qpos: ", result.qpos)
    # print("ctrl: ", hand.joint_positions_to_control(result.qpos))
    physics.bind(hand.joints).qpos = result.qpos
    physics.step()
    plt.imshow(render(physics))
    plt.show()
    # hand.set_position_control(
    #     physics,
    #     control=hand.joint_positions_to_control(result.qpos),
    # )

    # frames = []
    # framerate = 30
    # duration = 5.0
    # animate(physics, frames, duration=duration, framerate=framerate)
    # imageio.mimsave("temp/ik.mp4", frames, fps=framerate)


if __name__ == "__main__":
    main()
