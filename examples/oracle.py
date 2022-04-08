import time

import dm_env
import mujoco
import numpy as np
from absl import app
from absl import flags
from dm_control import mjcf
from dm_control.mujoco import wrapper

from shadow_hand import manipulation

flags.DEFINE_enum(
    "environment_name", "reach_state_dense", manipulation.ALL, "The environment name."
)
flags.DEFINE_integer("seed", None, "RNG seed.")
flags.DEFINE_integer("num_episodes", 1, "Number of episodes to run.")

FLAGS = flags.FLAGS


def render(physics: mjcf.Physics) -> np.ndarray:
    """Custom rendering with contact points and forces."""
    scene_option = wrapper.core.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    return physics.render(
        height=480, width=640, camera_id="front_close", scene_option=scene_option
    )


def main(_) -> None:
    env = manipulation.load(
        environment_name=FLAGS.environment_name,
        strip_singleton_obs_buffer_dim=True,
        seed=FLAGS.seed,
    )
    action_spec = env.action_spec()

    def oracle(timestep: dm_env.TimeStep) -> np.ndarray:
        del timestep  # Unused
        qpos = env.task._fingertips_initializer.qpos.copy()
        ctrl = env.task.hand.joint_positions_to_control(qpos)
        ctrl = ctrl.astype(action_spec.dtype)
        return ctrl

    for _ in range(FLAGS.num_episodes):
        frames = []
        timestep = env.reset()
        frames.append(render(env.physics))
        actions = []
        num_steps = 0
        returns = 0.0
        rewards = []
        episode_start = time.time()
        while True:
            action = oracle(timestep)
            actions.append(action)
            timestep = env.step(action)
            frames.append(render(env.physics))
            returns += timestep.reward
            rewards.append(timestep.reward)
            num_steps += 1
            if timestep.last():
                assert env.task.total_solves == 50, "Oracle failed to solve the task."
                break
        episode_time_ms = time.time() - episode_start

        # Print info.
        print(f"Episode time: {episode_time_ms:.2f} seconds.")
        print(f"{num_steps} steps taken.")
        print(f"Total reward: {returns}")
        print(f"Solves: {env.task.total_solves}")

    # import imageio
    # imageio.mimsave("temp/oracle_reach.mp4", frames, fps=30, quality=8)


if __name__ == "__main__":
    app.run(main)
