"""Placing fingertip locations at target sites using inverse kinematics."""

import dataclasses
import time
from typing import Dict, List, Optional

import dcargs
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

_SITE_COMPONENT_MAP = {
    "fftip_site": consts.Components.FF,
    "mftip_site": consts.Components.MF,
    "rftip_site": consts.Components.RF,
    "lftip_site": consts.Components.LF,
    "thtip_site": consts.Components.TH,
}


def render_scene(
    physics: mjcf.Physics, cam_id: str = "closeup", transparent: bool = False
) -> np.ndarray:
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    return physics.render(
        width=640, height=480, camera_id=cam_id, scene_option=scene_option
    )


def plot(image: np.ndarray, window: str = "") -> None:
    plt.figure(window)
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
            pixels = render_scene(physics, transparent=True)
            frames.append(pixels)
    return frames


def merge_solutions(
    physics: mjcf.Physics,
    joints: List[hints.MjcfElement],
    solutions: Dict[consts.Components, np.ndarray],
    solver: ik_solver.IKSolver,
) -> np.ndarray:
    joint_configuration = physics.bind(joints).qpos.copy()
    for finger, qpos in solutions.items():
        if finger == consts.Components.WR:
            joint_configuration[solver._wirst_joint_bindings.dofadr] = qpos
        else:
            joint_configuration[solver._joint_bindings[finger].dofadr] = qpos
    return joint_configuration


def _build_arena(name: str, disable_gravity: bool = True) -> Arena:
    arena = Arena(name)
    if disable_gravity:
        arena.mjcf_model.option.gravity = (0.0, 0.0, 0.0)
    arena.mjcf_model.size.nconmax = 1_000
    arena.mjcf_model.size.njmax = 2_000
    return arena


def _add_hand(arena: Arena):
    axis_angle = np.radians(180) * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
    quat = tr.axisangle_to_quat(axis_angle)
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
    return hand


@dataclasses.dataclass
class Args:
    seed: Optional[int] = None
    num_solves: int = 1
    linear_tol: float = 1e-4
    disable_plot: bool = False


def main(args: Args) -> None:
    if args.seed is not None:
        np.random.seed(args.seed)

    successes: int = 0
    for _ in range(args.num_solves):
        # Build the scene.
        arena = _build_arena("hand_ik")
        hand = _add_hand(arena)

        # Randomly sample a joint configuration.
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        joint_binding = physics.bind(hand.joints)
        rand_qpos = np.random.uniform(
            joint_binding.range[:, 0], joint_binding.range[:, 1]
        )
        rand_qpos[:2] = 0.0  # Disable wrist movement.

        # Set the configuration and query fingertip sites.
        physics.bind(hand.joints).qpos = rand_qpos
        target_positions = {}
        for fingertip_site in hand._fingertip_sites:
            target_positions[_SITE_COMPONENT_MAP[fingertip_site.name]] = physics.bind(
                fingertip_site
            ).xpos.copy()
        im_desired = render_scene(physics, transparent=False)

        # Add the target sites to the MJCF model for visualization purposes.
        for name, position in target_positions.items():
            arena.mjcf_model.worldbody.add(
                "site",
                name=f"{name}_target",
                type="sphere",
                pos=position,
                rgba="0 0 1 .7",
                size="0.001",
            )

        # Recreate physics instance since we changed the MJCF.
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        im_initial = render_scene(physics, transparent=False)

        solver = ik_solver.IKSolver(arena.mjcf_model, hand.mjcf_model.model)

        ik_start = time.time()
        joint_configurations = solver.solve(
            target_positions=target_positions,
            linear_tol=args.linear_tol,
            max_steps=100,
            early_stop=True,
            num_attempts=200,
            stop_on_first_successful_attempt=True,
        )
        ik_end = time.time()

        if joint_configurations is not None:
            print(f"Full IK solved in {ik_end - ik_start:.4f} seconds.")

            joint_configuration = merge_solutions(
                physics,
                hand.joints,
                joint_configurations,
                solver,
            )

            # Directly set joint angles and visualize.
            hand.set_joint_angles(physics, joint_configuration)
            physics.step()
            im_actual = render_scene(physics, transparent=True)

            if not args.disable_plot:
                _, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(im_initial)
                axes[0].set_title("Initial")
                axes[1].imshow(im_desired)
                axes[1].set_title("Desired")
                axes[2].imshow(im_actual)
                axes[2].set_title("Actual")
                for ax in axes:
                    ax.axis("off")
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.tight_layout()
                plt.show()

            successes += 1
        else:
            plot(im_desired, "failed")

    print(f"solve success rate: {successes}/{args.num_solves}.")


if __name__ == "__main__":
    main(dcargs.parse(Args, description=__doc__))
