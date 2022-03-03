# import numpy as np
# from dm_control import mjcf
# from dm_env import specs

# from shadow_hand import effector


# class RelativeEffector(effector.Effector):
#     def __init__(
#         self,
#         effector,
#     ) -> None:
#         self._effector = effector

#     def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
#         cur_command = physics.bind(self._effector._mujoco_effector._actuators).ctrl
#         delta_command = command - cur_command
#         self._effector.set_control(physics, delta_command)

#     def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
#         return self._effector.action_spec(physics)

#     def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
#         return self._effector.after_compile(mjcf_model)

#     def initialize_episode(
#         self, physics: mjcf.Physics, random_state: np.random.RandomState
#     ) -> None:
#         return self._effector.initialize_episode(physics, random_state)

#     @property
#     def prefix(self) -> str:
#         return self._effector.prefix
