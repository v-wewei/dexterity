"""Tasks involving hand finger reaching."""

import dataclasses
from typing import Dict, List

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.transformations import transformations as tr

from shadow_hand import effector
from shadow_hand import effectors
from shadow_hand import hints
from shadow_hand import task
from shadow_hand.manipulation import arenas
from shadow_hand.manipulation.shared import cameras
from shadow_hand.manipulation.shared import initializers
from shadow_hand.manipulation.shared import observations
from shadow_hand.manipulation.shared import registry
from shadow_hand.manipulation.shared import rewards
from shadow_hand.manipulation.shared import tags
from shadow_hand.manipulation.shared import workspaces
from shadow_hand.models.hands import fingered_hand
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts

# The position of the hand relative in the world frame, in meters.
_HAND_POS = (0, 0.2, 0.1)
# The orientation of the hand relative to the world frame.
_HAND_QUAT = tr.axisangle_to_quat(
    np.pi * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
)

_SITE_SIZE = 1e-2
_SITE_ALPHA = 0.1
_SITE_COLORS = (
    (1.0, 0.0, 0.0),  # Red.
    (0.0, 1.0, 0.0),  # Green.
    (0.0, 0.0, 1.0),  # Blue.
    (0.0, 1.0, 1.0),  # Cyan.
    (1.0, 0.0, 1.0),  # Magenta.
    (1.0, 1.0, 0.0),  # Yellow.
)
_TARGET_SIZE = 5e-3
_TARGET_ALPHA = 1.0

_STEPS_BEFORE_MOVING_TARGET: int = 5

# This is the threshold at which the distance between all the fingers of the hand and
# their target locations is considered small enough and the task is deemed successful.
_DISTANCE_TO_TARGET_THRESHOLD = 0.01  # 1 cm.

# Assign this color to the finger geoms if the finger is within the target threshold.
_THRESHOLD_COLOR = (0.0, 1.0, 0.0)

# Timestep of the physics simulation.
_PHYSICS_TIMESTEP: float = 0.01

# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP: float = 0.02

# The maximum number of consecutive solves until the task is terminated.
_MAX_SOLVES: int = 50

# The maximum allowed time for reaching the current target, in seconds.
_MAX_TIME_SINGLE_SOLVE: float = 5.0

# The maximum allowed time for the entire episode, in seconds.
_TIME_LIMIT = 35.0


class Reach(task.Task):
    """Move the fingers to desired goal positions."""

    def __init__(
        self,
        arena: composer.Arena,
        hand: fingered_hand.FingeredHand,
        hand_effector: effector.Effector,
        observable_settings: observations.ObservationSettings,
        use_dense_reward: bool,
        visualize_reward: bool,
        steps_before_moving_target: int = _STEPS_BEFORE_MOVING_TARGET,
        max_solves: int = _MAX_SOLVES,
        control_timestep: float = _CONTROL_TIMESTEP,
        physics_timestep: float = _PHYSICS_TIMESTEP,
    ) -> None:
        """Construct a new `Reach` task.

        Args:
            arena: The arena to use.
            hand: The hand to use.
            hand_effector: The effector to use for the hand.
            observable_settings: Settings for entity and task observables.
            use_dense_reward: Whether to use a dense reward.
            visualize_reward:
            steps_before_moving_target:
            control_timestep: The control timestep, in seconds.
            physics_timestep: The physics timestep, in seconds.
        """
        super().__init__(arena=arena, hand=hand, hand_effector=hand_effector)

        self._use_dense_reward = use_dense_reward
        self._visualize_reward = visualize_reward
        self._steps_before_moving_target = steps_before_moving_target
        self._max_solves = max_solves

        # Attach the hand to the arena.
        self._arena.attach_offset(hand, position=_HAND_POS, quaternion=_HAND_QUAT)

        self.set_timesteps(
            control_timestep=control_timestep,
            physics_timestep=physics_timestep,
        )

        # Create visible sites for the finger tips.
        for i, site in enumerate(self._hand.fingertip_sites):
            site.group = None  # Make the sites visible.
            site.size = (_SITE_SIZE,) * 3  # Increase their size.
            site.rgba = _SITE_COLORS[i] + (_SITE_ALPHA,)  # Change their color.

        # Create target sites for each fingertip.
        self._target_sites: List[hints.MjcfElement] = []
        for i, site in enumerate(self._hand.fingertip_sites):
            self._target_sites.append(
                workspaces.add_target_site(
                    body=arena.mjcf_model.worldbody,
                    radius=_TARGET_SIZE,
                    visible=True,
                    rgba=_SITE_COLORS[i] + (_TARGET_ALPHA,),
                    name=f"target_{site.name}",
                )
            )

        self._fingertips_initializer = initializers.FingertipPositionPlacer(
            target_sites=self._target_sites,
            hand=hand,
            ignore_self_collisions=False,
        )

        # Add camera observables.
        self._task_observables = cameras.add_camera_observables(
            arena, observable_settings, cameras.FRONT_CLOSE
        )

        # Add target positions as an observable.
        target_positions_observable = observable.Generic(self._get_target_positions)
        target_positions_observable.configure(
            **dataclasses.asdict(observable_settings.prop_pose),
        )
        self._task_observables["target_positions"] = target_positions_observable

    @property
    def task_observables(self) -> Dict[str, observable.Observable]:
        return self._task_observables

    @property
    def root_entity(self) -> composer.Entity:
        return self._arena

    @property
    def hand(self) -> fingered_hand.FingeredHand:
        return self._hand

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        super().initialize_episode(physics, random_state)

        # Sample a new goal position for each fingertip.
        self._fingertips_initializer(physics, random_state)
        self._total_solves = 0
        self._reward_step_counter = 0
        self._registered_solve = False
        self._exceeded_single_solve_time = False
        self._solve_start_time = physics.data.time

        # Save initial finger colors.
        if self._visualize_reward:
            self._init_finger_colors = {}
            for i, geoms in enumerate(consts.FINGER_GEOM_MAPPING.values()):
                elems = [
                    elem
                    for elem in self._hand.mjcf_model.find_all("geom")
                    if elem.name in geoms
                ]
                self._init_finger_colors[i] = (elems, physics.bind(elems).rgba)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        super().before_step(physics, action, random_state)

        if self._reward_step_counter >= self._steps_before_moving_target:
            self._fingertips_initializer(physics, random_state)
            self._reward_step_counter = 0
            self._registered_solve = False
            self._exceeded_single_solve_time = False
            self._solve_start_time = physics.data.time

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        super().after_step(physics, random_state)

        # Check if the fingers are close enough to their targets.
        goal_pos = self._get_target_positions(physics).reshape(-1, 3)
        cur_pos = self._get_fingertip_positions(physics).reshape(-1, 3)
        self._distance = np.linalg.norm(goal_pos - cur_pos, axis=1)
        if np.all(self._distance <= _DISTANCE_TO_TARGET_THRESHOLD):
            self._reward_step_counter += 1
            if not self._registered_solve:
                self._total_solves += 1
                self._registered_solve = True
        else:
            if physics.data.time - self._solve_start_time > _MAX_TIME_SINGLE_SOLVE:
                self._exceeded_single_solve_time = True

        # If they are close enough, change their color.
        if self._visualize_reward:
            self._maybe_color_fingers(physics)

    def get_reward(self, physics: mjcf.Physics) -> float:
        del physics  # Unused.
        if self._use_dense_reward:
            # Dense reward:
            # For each fingertip that is close enough to the target, we reward it with
            # a +1.
            # Otherwise, we reward it using a distance function D(a, b) which is defined
            # as the tanh over the Euclidean distance between a and b, which decays to
            # 0.05 as the distance reaches 0.1.
            return np.mean(
                np.where(
                    self._distance <= _DISTANCE_TO_TARGET_THRESHOLD,
                    1.0,
                    [1.0 - rewards.tanh_squared(d, margin=0.1) for d in self._distance],
                )
            )
        # Sparse reward:
        # For each fingertip that is close enough to the target, we reward it with a 1.
        # Otherwise, no reward is given.
        # We then return the average over all fingertips.
        # So if all fingertips are within the target, the net reward is 1.0.
        return np.mean(
            np.where(self._distance <= _DISTANCE_TO_TARGET_THRESHOLD, 1.0, 0.0)
        )

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        del physics  # Unused.
        if self._total_solves >= self._max_solves or self._exceeded_single_solve_time:
            return True
        return False

    @property
    def total_solves(self) -> int:
        return self._total_solves

    @property
    def time_limit(self) -> float:
        return _TIME_LIMIT

    # Helper methods.

    def _get_target_positions(self, physics: mjcf.Physics) -> np.ndarray:
        """Returns the desired fingertip Cartesian positions in the world frame.

        The returned array is of shape (15,).
        """
        return np.array(physics.bind(self._target_sites).xpos).ravel()

    def _get_fingertip_positions(self, physics: mjcf.Physics) -> np.ndarray:
        """Returns the current fingertip Cartesian positions in the world frame.

        The returned array is of shape (15,).
        """
        return np.array(physics.bind(self._hand.fingertip_sites).xpos).ravel()

    def _maybe_color_fingers(self, physics: mjcf.Physics) -> None:
        for i, distance in enumerate(self._distance):
            elems, rgba = self._init_finger_colors[i]
            if distance <= _DISTANCE_TO_TARGET_THRESHOLD:
                physics.bind(elems).rgba = _THRESHOLD_COLOR + (1.0,)
            else:
                physics.bind(elems).rgba = rgba


def reach_task(
    observation_set: observations.ObservationSet,
    use_dense_reward: bool,
    visualize_reward: bool = True,
) -> composer.Task:
    """Configure and instantiate a `Reach` task."""
    arena = arenas.Standard()

    hand = shadow_hand_e.ShadowHandSeriesE(
        observable_options=observations.make_options(
            observation_set.value,
            observations.HAND_OBSERVABLES,
        ),
    )

    hand_effector = effectors.HandEffector(hand=hand, hand_name=hand.name)

    return Reach(
        arena=arena,
        hand=hand,
        hand_effector=hand_effector,
        observable_settings=observation_set.value,
        use_dense_reward=use_dense_reward,
        visualize_reward=visualize_reward,
    )


@registry.add(tags.STATE, tags.DENSE)
def reach_state_dense() -> composer.Task:
    """Reach task with full state observations and dense reward."""
    return reach_task(
        observation_set=observations.ObservationSet.STATE_ONLY,
        use_dense_reward=True,
        visualize_reward=True,
    )


@registry.add(tags.STATE, tags.SPARSE)
def reach_state_sparse() -> composer.Task:
    """Reach task with full state observations and sparse reward."""
    return reach_task(
        observation_set=observations.ObservationSet.STATE_ONLY,
        use_dense_reward=False,
        visualize_reward=True,
    )
