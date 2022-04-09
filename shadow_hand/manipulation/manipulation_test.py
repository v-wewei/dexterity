"""Tests for manipulation."""

import collections
from typing import Mapping

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from shadow_hand import manipulation

_SEED = 12345
_NUM_EPISODES = 5
_NUM_STEPS_PER_EPISODE = 10
_DOMAINS_AND_TASKS = [
    dict(domain=domain, task=task) for domain, task in manipulation.ALL_TASKS
]


class ManipulationTest(parameterized.TestCase):
    """Tests run on all the registered tasks."""

    @parameterized.parameters(_DOMAINS_AND_TASKS)
    def test_task_runs(self, domain: str, task: str) -> None:
        """Tests that the environment runs and is coherent with its specs."""
        env = manipulation.load(domain, task, seed=_SEED)
        random_state = np.random.RandomState(_SEED)

        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        self.assertTrue(np.all(np.isfinite(action_spec.minimum)))
        self.assertTrue(np.all(np.isfinite(action_spec.maximum)))

        # Run a partial episode, check observations, rewards, discount.
        for _ in range(_NUM_EPISODES):
            time_step = env.reset()
            for _ in range(_NUM_STEPS_PER_EPISODE):
                self._validate_observation(time_step.observation, observation_spec)
                if time_step.first():
                    self.assertIsNone(time_step.reward)
                    self.assertIsNone(time_step.discount)
                else:
                    self._validate_discount(time_step.discount)
                action = random_state.uniform(action_spec.minimum, action_spec.maximum)
                action = action.astype(action_spec.dtype)
                time_step = env.step(action)

    # Helper methods.

    def _validate_observation(
        self,
        observation: Mapping[str, np.ndarray],
        observation_spec: collections.OrderedDict,
    ) -> None:
        self.assertEqual(list(observation.keys()), list(observation_spec.keys()))
        for name, array_spec in observation_spec.items():
            array_spec.validate(observation[name])

    def _validate_discount(self, discount: float) -> None:
        self.assertIsInstance(discount, float)
        self.assertBetween(discount, 0.0, 1.0)


if __name__ == "__main__":
    absltest.main()
