# Manipulation Task Library

This library facilitates the creation of tasks involving uni and bimanual dexterous manipulation. New users are encouraged to browse the available [tasks](tasks/) to get a feel for how to weave the library components together to design RL environments.

<p float="left">
  <img src="../../assets/reach.png" height="200">
  <img src="../../assets/cube.png" height="200">
</p>

## Terminology

We follow a similar design pattern to dm_control's [`locomotion`](https://github.com/deepmind/dm_control/tree/main/dm_control/locomotion) library. In our case, an environment consists of one or more **hands** performing a **task** (potentially involving **props**) in an **arena**.

- **hand**: An instance of [`dexterity.models.hands.DexterousHand`](../models/hands/dexterous_hand.py#L75); refers to the robotic hand that we wish to control. See [`dexterity.models.hands`](../models/hands/) for a list of available hand models.
- **arena**: An instance of [`dexterity.models.arenas.Arena`](../models/arenas/arena.py#L12); refers to the surroundings in which the hand(s) and other objects exist. See [`dexterity.models.arenas`](../models/arenas/) for a list of available arenas.
- **prop**: An instance of [`composer.Entity`](https://github.com/deepmind/dm_control/blob/main/dm_control/composer/entity.py); refers to a non-actuated object in the arena. For example this could be an object that the hand has to manipulate or it could be a table on which other props are placed.
- **task**: An instance of [`task.Task`](../task.py#L17) or [`task.GoalTask`](../task.py#L112); refers to the specification of observations and rewards along with initialization and termination logic.

## Quickstart

```bash
# Run --help to get a list of command line flags.
python explore.py
```

This will print out all the available tasks. You can then select a specific one and interact with it via the [`dm_control.viewer`](https://github.com/deepmind/dm_control/blob/main/dm_control/viewer/README.md). Additionally, here's a code snippet to get your feet wet:

```python
import numpy as np
from dexterity import manipulation

# Load a task.
env = manipulation.load(domain_name="reach", task_name="state_dense")

# Iterate over all task sets.
for domain_name, task_name in manipulation.ALL_TASKS:
    print(f"{domain_name} - {task_name}")

# Print out the environment's specs:
action_spec = env.action_spec()
observation_spec = env.observation_spec()
print(action_spec, observation_spec)

# Step through an episode and print out reward, discount and observation.
timestep = env.reset()
while not timestep.last():
    action = np.random.uniform(action_spec.minimum, action_spec.maximum)
    timestep = env.step(action)
    print(timestep.reward, timestep.discount, timestep.observation)
```

## Illustration Video

Below is a video of the [`Reach`](tasks/reach.py) task being solved by an oracle.

[![Reach Task Oracle Rollout](https://img.youtube.com/vi/IULfrF6ya1E/0.jpg)](https://youtu.be/IULfrF6ya1E)
