import dataclasses
from typing import Mapping


@dataclasses.dataclass(frozen=True)
class Reward:
    value: float
    weight: float


@dataclasses.dataclass(frozen=True)
class ShapedReward:
    """Convenience class for storing and computing shaped rewards."""

    rewards: Mapping[str, Reward] = dataclasses.field(default_factory=dict)

    def add(self, name: str, value: float, weight: float) -> "ShapedReward":
        return dataclasses.replace(
            self,
            rewards={**self.rewards, name: Reward(value=value, weight=weight)},
        )

    @property
    def weighted_average(self) -> float:
        if not self.rewards:
            raise ValueError("No rewards to average.")

        return sum([reward.value * reward.weight for reward in self.rewards.values()])
