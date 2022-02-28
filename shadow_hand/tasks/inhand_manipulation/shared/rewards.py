import dataclasses
from typing import Mapping


@dataclasses.dataclass(frozen=True)
class Reward:
    value: float
    weight: float


def weight_average(rewards: Mapping[str, Reward]) -> float:
    """Compute a weighted average shaped reward components."""
    return sum([reward.value * reward.weight for reward in rewards.values()])
