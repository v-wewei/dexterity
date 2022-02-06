import numpy as np
from dm_control import mjcf, mujoco
from dm_control.utils import inverse_kinematics
from dm_robotics.transformations import transformations as tr
from matplotlib import pyplot as plt

from shadow_hand.models.arenas.empty import Arena
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts


def render(physics: mjcf.Physics, cam_id: str = "fixed_viewer") -> np.ndarray:
    scene_option = mujoco.wrapper.core.MjvOption()
    # scene_option.frame = enums.mjtFrame.mjFRAME_GEOM
    return physics.render(
        width=640, height=480, camera_id=cam_id, scene_option=scene_option
    )


def main() -> None:
    # Build the arena.
    arena = Arena("hand_arena")

    target_pos = (-0.01, -0.5, 0.24)
    arena.mjcf_model.worldbody.add(
        "site",
        type="sphere",
        pos=target_pos,
        rgba="0 1 0 0.3",
        size="0.01",
    )

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

    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    axis_angle = np.pi * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
    hand.set_pose(physics, quaternion=tr.axisangle_to_quat(axis_angle))
    physics.step()

    plt.imshow(render(physics))
    plt.show()

    site_name = hand._fingertip_sites[1].full_identifier
    print(site_name)
    max_ik_attempts = 10

    # Just first finger and wrist joints.
    joint_names = []
    for joint in hand.joints:
        if "mf" in joint.name.lower():
            joint_names.append(joint.full_identifier)
        if "wr" in joint.name.lower():
            joint_names.append(joint.full_identifier)
    print(joint_names)

    for i in range(max_ik_attempts):
        result = inverse_kinematics.qpos_from_site_pose(
            physics=physics,
            site_name=site_name,
            target_pos=target_pos,
            joint_names=joint_names,
            inplace=True,
            tol=1e-2,
            rot_weight=1.0,
            # regularization_threshold=0.1,
            # regularization_strength=3e-2,
            # max_update_norm=2.0,
            # progress_thresh=20.0,
            max_steps=100,
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

    physics.step()
    plt.imshow(render(physics))
    plt.show()


if __name__ == "__main__":
    main()
