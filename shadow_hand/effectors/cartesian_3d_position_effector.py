# from typing import Mapping

# import numpy as np
# from dm_control import mjcf
# from dm_env import specs

# from shadow_hand import effector
# from shadow_hand.models.hands import shadow_hand_e_constants as consts


# class Cartesian3dFingertipPositionEffector(effector.Effector):
#     """A Cartesian 3D fingertip position effector interface for a dexterous hand.

#     This effector takes in 3D Cartesian positions for the fingertips of the hand and
#     maps them to joint positions which are fed to the underlying MuJoCo position
#     actuators.
#     """

#     def __init__(
#         self,
#         hand_name: str,
#         joint_position_effector: effector.Effector,
#     ) -> None:
#         self._prefix = f"{hand_name}_cartesian_3d_fingertip_position_effector"
#         self._joint_position_effector = joint_position_effector

#     def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
#         # Construct the Jacobian pseudoinverse mapper.
#         self._mapper = None

#     def initialize_episode(
#         self, physics: mjcf.Physics, random_state: np.random.RandomState
#     ) -> None:
#         self._joint_position_effector.initialize_episode(physics, random_state)

#     def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
#         pass

#     @property
#     def prefix(self) -> str:
#         return self._prefix
