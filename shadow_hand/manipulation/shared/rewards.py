import dataclasses
from typing import Mapping

import numpy as np


@dataclasses.dataclass(frozen=True)
class Reward:
    value: float
    weight: float


def weighted_average(rewards: Mapping[str, Reward]) -> float:
    """Computes a weighted average of shaped reward components."""
    return sum([reward.value * reward.weight for reward in rewards.values()])


def tanh_squared(x: np.ndarray, margin: float, loss_at_margin: float = 0.95):
    if not margin > 0:
        raise ValueError("`margin` must be positive.")
    if not 0.0 < loss_at_margin < 1.0:
        raise ValueError("`loss_at_margin` must be between 0 and 1.")

    error = np.linalg.norm(x)
    # Compute weight such that at the margin tanh(w * error) = loss_at_margin
    w = np.arctanh(np.sqrt(loss_at_margin)) / margin
    s = np.tanh(w * error)
    return s * s
