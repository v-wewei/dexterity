import dataclasses
from typing import Mapping


@dataclasses.dataclass(frozen=True)
class Reward:
    value: float
    weight: float


def weighted_average(rewards: Mapping[str, Reward]) -> float:
    """Computes a weighted average of shaped reward components."""
    return sum([reward.value * reward.weight for reward in rewards.values()])
