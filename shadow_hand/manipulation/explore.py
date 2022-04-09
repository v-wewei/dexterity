"""A standalone app for visualizing manipulation tasks."""

from typing import Sequence

from absl import app
from absl import flags
from dm_control import viewer

from shadow_hand import manipulation

_ALL_NAMES = [".".join(domain_task) for domain_task in manipulation.ALL_TASKS]

flags.DEFINE_enum(
    "environment_name",
    None,
    _ALL_NAMES,
    "Optional 'domain_name.task_name' pair specifying the environment to load.",
)
flags.DEFINE_integer("seed", None, "RNG seed.")
flags.DEFINE_bool("timeout", True, "Whether episodes should have a time limit.")

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
    if FLAGS.environment_name is None:
        print("\n ".join(["Available environments:"] + _ALL_NAMES))
        environment_name = prompt_environment_name(_ALL_NAMES)
    else:
        environment_name = FLAGS.environment_name

    index = _ALL_NAMES.index(environment_name)
    domain_name, task_name = manipulation.ALL_TASKS[index]

    env = manipulation.load(
        domain_name=domain_name,
        task_name=task_name,
        seed=FLAGS.seed,
        time_limit=float("inf") if not FLAGS.timeout else None,
    )

    # Print entity and task observables.
    timestep = env.reset()
    for k, v in timestep.observation.items():
        print(f"{k}: {v.shape}")

    viewer.launch(env)


if __name__ == "__main__":
    app.run(main)
