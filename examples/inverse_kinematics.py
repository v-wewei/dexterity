import time
from typing import Dict, List, Optional

import imageio
import numpy as np
from dm_control import mjcf, mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_robotics.transformations import transformations as tr
from matplotlib import pyplot as plt

from shadow_hand import hints
from shadow_hand.ik import ik_solver
from shadow_hand.models.arenas.empty import Arena
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts

TARGET_POSITIONS: Dict[consts.Components, np.ndarray] = {
    consts.Components.LF: np.array([0.03, -0.38, 0.16]),
    consts.Components.RF: np.array([0.01, -0.38, 0.16]),
    consts.Components.MF: np.array([-0.01, -0.38, 0.16]),
    consts.Components.FF: np.array([-0.03, -0.38, 0.16]),
    consts.Components.TH: np.array([-0.03, -0.345, 0.13]),
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


def merge_solutions(
    physics: mjcf.Physics,
    joints: List[hints.MjcfElement],
    solutions: Dict[consts.Components, Optional[np.ndarray]],
    solver: ik_solver.IKSolver,
) -> np.ndarray:
    joint_configuration = physics.bind(joints).qpos.copy()
    for finger, qpos in solutions.items():
        if finger == consts.Components.WR:
            if qpos is not None:
                joint_configuration[solver._wirst_joint_bindings.dofadr] = qpos
        else:
            if qpos is not None:
                joint_configuration[solver._joint_bindings[finger].dofadr] = qpos
    return joint_configuration


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

    solver = ik_solver.IKSolver(arena.mjcf_model, hand.mjcf_model.model)

    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    plot(render(physics, transparent=False))

    ik_start = time.time()
    solutions = solver.solve(
        target_positions=TARGET_POSITIONS,
        linear_tol=1e-6,
        max_steps=100,
        early_stop=True,
        num_attempts=30,
        stop_on_first_successful_attempt=False,
    )
    print(f"Full IK solved in {time.time() - ik_start:.4f} seconds.")

    joint_configuration = merge_solutions(
        physics,
        hand.joints,
        solutions,
        solver,
    )

    # Command the actuators and animate.
    ctrl = hand.joint_positions_to_control(joint_configuration)
    hand.set_position_control(physics, ctrl)
    frames = animate(physics, duration=5.0)
    imageio.mimsave("temp/inverse_kinematics.mp4", frames, fps=30)

    # Directly set joint angles and visualize.
    hand.set_joint_angles(physics, joint_configuration)
    physics.step()
    im0 = render(physics, transparent=False)
    im1 = render(physics, transparent=True)
    plot(np.hstack([im0, im1]))


if __name__ == "__main__":
    main()
