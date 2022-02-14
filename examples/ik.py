import time
from typing import Dict, List, Tuple

import imageio
import numpy as np
from dm_control import mjcf, mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_robotics.transformations import transformations as tr
from matplotlib import pyplot as plt

from shadow_hand.models.arenas.empty import Arena
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts
from shadow_hand.utils import ik_solver

TARGET_POSITIONS: Dict[consts.Components, Tuple[float, float, float]] = {
    consts.Components.LF: (0.03, -0.38, 0.16),
    consts.Components.RF: (0.01, -0.38, 0.16),
    consts.Components.MF: (-0.01, -0.38, 0.16),
    consts.Components.FF: (-0.03, -0.38, 0.16),
    consts.Components.TH: (-0.03, -0.345, 0.13),
}


def render(
    physics: mjcf.Physics, cam_id: str = "fixed_viewer1", transparent: bool = False
) -> np.ndarray:
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    return physics.render(
        width=640, height=480, camera_id=cam_id, scene_option=scene_option
    )


def plot(image: np.ndarray) -> None:
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()


def animate(
    physics: mjcf.Physics,
    duration: float = 2.0,
    framerate: float = 30,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    while physics.data.time < duration:
        physics.step()
        if len(frames) < physics.data.time * framerate:
            pixels = render(physics, transparent=True)
            frames.append(pixels)
    return frames


def main() -> None:
    # Build the arena.
    arena = Arena("hand_arena")
    arena.mjcf_model.option.gravity = (0.0, 0.0, 0.0)  # Disable gravity.
    arena.mjcf_model.size.nconmax = 1_000
    arena.mjcf_model.size.njmax = 2_000

    # Create sites for target fingertip positions.
    target_site_elems = []
    for name, position in TARGET_POSITIONS.items():
        site_elem = arena.mjcf_model.worldbody.add(
            "site",
            name=f"{name}_target",
            type="sphere",
            pos=position,
            rgba="0 0 1 .5",
            size="0.001",
        )
        target_site_elems.append(site_elem)

    axis_angle = np.radians(180) * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
    quat = tr.axisangle_to_quat(axis_angle)

    # Load the hand and add it to the arena.
    attachment_site = arena.mjcf_model.worldbody.add(
        "site",
        type="sphere",
        pos=[0, 0, 0.1],
        quat=quat,
        rgba="0 0 0 0",
        size="0.01",
    )
    hand = shadow_hand_e.ShadowHandSeriesE(actuation=consts.Actuation.POSITION)
    arena.attach(hand, attachment_site)

    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    plot(render(physics, transparent=False))

    ik_solvers = {}
    finger_controllable_joints = {}
    for finger in [
        consts.Components.TH,
        consts.Components.FF,
        consts.Components.MF,
        consts.Components.RF,
        consts.Components.LF,
    ]:
        # Get elem associated with fingertip.
        fingertip_name = finger.name.lower() + "tip"
        fingertip_site_name = f"{hand.mjcf_model.model}/{fingertip_name}_site"
        fingertip_site_elem = arena.mjcf_model.find("site", fingertip_site_name)
        assert fingertip_site_elem is not None

        # Get controllable joints for the hand given finger pose.
        finger_joints = consts.JOINT_GROUP[finger]
        joints = finger_joints
        controllable_joints = []
        for joint, joint_elem in hand._joint_elem_mapping.items():
            if joint in joints:
                controllable_joints.append(joint_elem)
        assert len(controllable_joints) == len(joints)

        physics_joints = physics.bind(controllable_joints)

        ik_solvers[finger] = ik_solver.IKSolver(
            model=arena.mjcf_model,
            controllable_joints=controllable_joints,
            element=fingertip_site_elem,
        )
        finger_controllable_joints[finger] = tuple(controllable_joints)

    joint_positions = {}
    for finger in [
        consts.Components.TH,
        consts.Components.FF,
        consts.Components.MF,
        consts.Components.RF,
        consts.Components.LF,
    ]:
        # Solve.
        target_position = np.array(TARGET_POSITIONS[finger])

        tic = time.time()
        qpos = ik_solvers[finger].solve(
            target_position=target_position,
            max_steps=100,
            num_attempts=30,
            stop_on_first_successful_attempt=True,
            linear_tol=1e-5,
        )
        print(f"Solved {finger} IK in {time.time() - tic:.4f} seconds.")
        joint_positions[finger] = qpos

    # Command the actuators.
    joint_angles = np.zeros(len(hand.joints))
    for finger, qpos in joint_positions.items():
        if qpos is not None:
            physics_joints = physics.bind(finger_controllable_joints[finger])
            joint_angles[physics_joints.dofadr] = qpos
    ctrl = hand.joint_positions_to_control(joint_angles)
    hand.set_position_control(physics, ctrl)
    frames = animate(physics, duration=5.0)
    imageio.mimsave("temp/ik.mp4", frames, fps=30)

    # Directly set joint angles.
    joint_angles = np.zeros(len(hand.joints))
    for finger, qpos in joint_positions.items():
        if qpos is not None:
            physics_joints = physics.bind(finger_controllable_joints[finger])
            joint_angles[physics_joints.dofadr] = qpos
    hand.set_joint_angles(physics, joint_angles)
    physics.step()
    im0 = render(physics, transparent=False)
    im1 = render(physics, transparent=True)
    plot(np.hstack([im0, im1]))


if __name__ == "__main__":
    main()
