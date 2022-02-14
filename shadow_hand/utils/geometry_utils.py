import numpy as np
from dm_robotics.transformations import transformations as tr


def get_orientation_error(to_quat: np.ndarray, from_quat: np.ndarray) -> np.ndarray:
    """Returns error between the two quaternions."""
    err_quat = tr.quat_diff_active(from_quat, to_quat)
    return tr.quat_to_axisangle(err_quat)
