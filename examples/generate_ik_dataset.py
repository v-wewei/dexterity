"""Generates a dataset of joint configuration / fingertip position pairs."""

import pathlib
import pickle
import random

import numpy as np
import tqdm
from absl import app
from absl import flags
from dm_control import mjcf
from dm_robotics.transformations import transformations as tr

from shadow_hand.manipulation.arenas import Standard
from shadow_hand.manipulation.shared import cameras
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.utils import mujoco_collisions

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("dataset_size", 10_000, "The size of the dataset.")
flags.DEFINE_string("save_dir", None, "Where to dump the dataset.")

FLAGS = flags.FLAGS


def main(_) -> None:
    # Seed the RNG.
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    # Build the scene.
    arena = Standard()
    axis_angle = np.radians(180) * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
    quat = tr.axisangle_to_quat(axis_angle)
    hand = shadow_hand_e.ShadowHandSeriesE()
    arena.attach_offset(hand, position=(0, 0.2, 0.1), quaternion=quat)
    arena.mjcf_model.worldbody.add(
        "camera",
        name=cameras.FRONT_CLOSE.name,
        pos=cameras.FRONT_CLOSE.pos,
        xyaxes=cameras.FRONT_CLOSE.xyaxes,
    )
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    binding = physics.bind(hand.joints)

    # Generate!
    dataset = []
    for _ in tqdm.tqdm(range(FLAGS.dataset_size)):
        # Sample a collision-free configuration.
        while True:
            qpos = np.random.uniform(binding.range[:, 0], binding.range[:, 1])
            hand.set_joint_angles(physics, qpos)
            physics.forward()
            if not mujoco_collisions.has_self_collision(physics, hand.name):
                break

        # Forward kinematics to compute fingertip positions, in the world frame.
        finger_positions = {}
        for finger, fingertip_site in hand._fingertip_site_elem_mapping.items():
            finger_positions[finger] = physics.bind(fingertip_site).xpos.copy()

        # Store.
        dataset.append(
            {
                "qpos": qpos,
                "finger_positions": finger_positions,
            }
        )

    # Dump.
    save_dir = pathlib.Path(FLAGS.save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    with open(save_dir / "ik_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    flags.mark_flag_as_required("save_dir")
    app.run(main)
