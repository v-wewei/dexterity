from typing import Optional

import numpy as np
from dm_robotics.transformations import transformations as tr


def l2_normalize(
    x: np.ndarray,
    axis: Optional[int] = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """L2 normalize an array with numerical stability."""
    square_sum = np.sum(np.square(x), axis=axis, keepdims=True)  # type: ignore
    x_inv_norm = 1.0 / np.sqrt(np.maximum(square_sum, epsilon))
    return x * x_inv_norm


def get_orientation_error(to_quat: np.ndarray, from_quat: np.ndarray) -> np.ndarray:
    """Returns the error between the two quaternions as an axis-angle."""
    err_quat = tr.quat_diff_active(from_quat, to_quat)
    # NOTE(kevin): The norm of this axis-angle is the scalar valued angular error.
    return tr.quat_to_axisangle(err_quat)


def quaternion_equal(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Returns True if two quaternions are equal."""
    actual = np.array(actual)
    expected = np.array(expected)
    return np.allclose(actual, expected) or np.allclose(actual, expected * -1)
