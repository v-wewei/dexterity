from typing import Optional, Tuple

from dm_control import composer as _composer

from shadow_hand.manipulation.shared import registry as _registry
from shadow_hand.manipulation.tasks import reach as _reach
from shadow_hand.manipulation.tasks import reorient as _reorient
from shadow_hand.task import Task
from shadow_hand.utils import mujoco_collisions

_registry.done_importing_tasks()

ALL: Tuple[str, ...] = tuple(_registry.get_all_names())


def get_environments_by_tag(tag: str) -> Tuple[str, ...]:
    """Returns the names of all environments matching a given tag."""
    return tuple(_registry.get_names_by_tag(tag))


def load(environment_name: str, seed: Optional[int] = None) -> _composer.Environment:
    # Build the task.
    task: Task = _registry.get_constructor(environment_name)()

    # Ensure MuJoCo will not check for collisions between geoms that can never collide.
    mujoco_collisions.exclude_bodies_based_on_contype_conaffinity(
        task.root_entity.mjcf_model
    )

    return _composer.Environment(task=task, random_state=seed)
