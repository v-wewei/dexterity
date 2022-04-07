"""A standalone app for visualizing manipulation tasks."""

from typing import Sequence

import dm_env
import numpy as np
from absl import app
from absl import flags
from dm_control import viewer

from shadow_hand import manipulation

flags.DEFINE_enum(
    "environment_name",
    None,
    manipulation.ALL,
    "Optional name of an environment to load. If unspecified a prompt will appear to "
    "select one.",
)
flags.DEFINE_integer("seed", None, "RNG seed.")
flags.DEFINE_boolean("no_policy", False, "If toggled, disables the random policy.")

FLAGS = flags.FLAGS


def prompt_environment_name(values: Sequence[str]) -> str:
    environment_name = None
    while not environment_name:
        environment_name = input("Please enter the environment name: ")
        if not environment_name or environment_name not in values:
            print(f"'{environment_name}' is not a valid environment name.")
            environment_name = None
    return environment_name


def main(_) -> None:
    all_names = list(manipulation.ALL)

    if FLAGS.environment_name is None:
        print("\n ".join(["Available environments:"] + all_names))
        environment_name = prompt_environment_name(all_names)
    else:
        environment_name = FLAGS.environment_name

    env = manipulation.load(environment_name=environment_name, seed=FLAGS.seed)
    action_spec = env.action_spec()

    # Print entity and task observables.
    timestep = env.reset()
    for k, v in timestep.observation.items():
        print(f"{k}: {v.shape}")

    def oracle(timestep: dm_env.TimeStep) -> np.ndarray:
        del timestep  # Unused
        qpos = env.task._fingertips_initializer.qpos.copy()
        ctrl = env.task.hand.joint_positions_to_control(qpos)
        ctrl = ctrl.astype(action_spec.dtype)
        return ctrl

    viewer.launch(env, policy=None if FLAGS.no_policy else oracle)


if __name__ == "__main__":
    app.run(main)
