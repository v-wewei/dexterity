# """Placing fingertip locations at target sites using inverse kinematics."""

# from typing import Dict, List

# import imageio
# import numpy as np
# from dm_control import mjcf, mujoco
# from dm_control.mujoco.wrapper.mjbindings import enums
# from dm_robotics.transformations import transformations as tr
# from matplotlib import pyplot as plt

# from shadow_hand import hints
# from shadow_hand.ik import ik_solver
# from shadow_hand.models.arenas.empty import Arena
# from shadow_hand.models.hands import shadow_hand_e
# from shadow_hand.models.hands import shadow_hand_e_constants as consts


# def render_scene(
#     physics: mjcf.Physics, cam_id: str = "fixed_viewer1", transparent: bool = False
# ) -> np.ndarray:
#     scene_option = mujoco.wrapper.core.MjvOption()
#     scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
#     return physics.render(
#         width=640, height=480, camera_id=cam_id, scene_option=scene_option
#     )


# def plot(image: np.ndarray, window: str = "") -> None:
#     plt.figure(window)
#     plt.imshow(image)
#     plt.axis("off")
#     plt.show()
#     plt.close()


# def animate(
#     physics: mjcf.Physics,
#     duration: float = 2.0,
#     framerate: float = 30,
# ) -> List[np.ndarray]:
#     frames: List[np.ndarray] = []
#     while physics.data.time < duration:
#         physics.step()
#         if len(frames) < physics.data.time * framerate:
#             pixels = render_scene(physics, transparent=True)
#             frames.append(pixels)
#     return frames


# def merge_solutions(
#     physics: mjcf.Physics,
#     joints: List[hints.MjcfElement],
#     solutions: Dict[consts.Components, np.ndarray],
#     solver: ik_solver.IKSolver,
# ) -> np.ndarray:
#     joint_configuration = physics.bind(joints).qpos.copy()
#     for finger, qpos in solutions.items():
#         if finger == consts.Components.WR:
#             joint_configuration[solver._wirst_joint_bindings.dofadr] = qpos
#         else:
#             joint_configuration[solver._joint_bindings[finger].dofadr] = qpos
#     return joint_configuration


# def _build_arena(name: str, disable_gravity: bool = True) -> Arena:
#     arena = Arena(name)
#     if disable_gravity:
#         arena.mjcf_model.option.gravity = (0.0, 0.0, 0.0)
#     arena.mjcf_model.size.nconmax = 1_000
#     arena.mjcf_model.size.njmax = 2_000
#     return arena


# def _add_hand(arena: Arena) -> shadow_hand_e.ShadowHandSeriesE:
#     axis_angle = np.radians(180) * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
#     quat = tr.axisangle_to_quat(axis_angle)
#     attachment_site = arena.mjcf_model.worldbody.add(
#         "site",
#         type="sphere",
#         pos=[0, 0, 0.1],
#         quat=quat,
#         rgba="0 0 0 0",
#         size="0.01",
#     )
#     hand = shadow_hand_e.ShadowHandSeriesE(actuation=consts.Actuation.POSITION)
#     arena.attach(hand, attachment_site)
#     return hand


# def add_ball(arena: Arena):
#     ball = arena.mjcf_model.worldbody.add(
#         "body",
#         name="ball",
#         pos="0.0 0.15 0.2",
#     )
#     ball.add("freejoint")
#     ball.add(
#         "geom",
#         type="sphere",
#         size="0.028",
#         rgba="0.5 0.5 0.5 1",
#         mass="0.043",
#         condim="4",
#     )
#     return ball


# def main() -> None:
#     # Build the scene.
#     arena = _build_arena("hand_ik")
#     hand = _add_hand(arena)
#     ball = add_ball(arena)

#     # Randomly sample a joint configuration.
#     physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

#     plot(render_scene(physics))

#     # frames = animate(physics, duration=5.0)
#     # imageio.mimsave("temp/ball.mp4", frames, fps=30)


# if __name__ == "__main__":
#     main()
