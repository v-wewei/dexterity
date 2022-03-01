"""A standalone app for visualizing in-hand manipulation tasks."""

import functools
from typing import Sequence

from absl import app
from absl import flags
from dm_control import viewer

from shadow_hand.tasks import inhand_manipulation

flags.DEFINE_enum(
    "environment_name",
    None,
    inhand_manipulation.ALL,
    "Optional name of an environment to load. If unspecified a prompt will appear to "
    "select one.",
)

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
    all_names = list(inhand_manipulation.ALL)

    if FLAGS.environment_name is None:
        print("\n ".join(["Available environments:"] + all_names))
        environment_name = prompt_environment_name(all_names)
    else:
        environment_name = FLAGS.environment_name

    loader = functools.partial(
        inhand_manipulation.load, environment_name=environment_name
    )
    viewer.launch(loader)


if __name__ == "__main__":
    app.run(main)
