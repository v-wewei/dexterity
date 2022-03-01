# Logbook

## 01/03/2022

Questions I've been thinking about:

### Actuator Specs

Should actuators for the shadow hand be scaled to the [-1, 1] range? The actual joint ranges aren't symmetric about 0, e.g. the fingers can't move behind the palm so if the actuator range is scaled, then the zero vector doesn't map to a flat hand but to the midrange of the joints which is probably not what we want.

### Termination and Discount

From: [dm_env](https://github.com/deepmind/dm_env/blob/master/docs/index.md):

* A sequence, aka an episode, consists of a series of `TimeStep`s returned by consecutive calls to `step()`.
* A prediction is the discounted sum of future rewards that we wish to predict.
* The discount does *not* determine when a sequence ends. The discount may be 0 in the middle of a sequence and ≥0 at the end of a sequence.

When a prediction ends at the end of a sequence, it is like a finite-horizon RL task. When a prediction does not end at the end of a sequence, it is like an infinite-horizon RL task.

* dm_control.suite
    * physics timestep: 0.01
    * time limit: 10 seconds
    * steps taken = time limit / physics timestep = 10 / 0.01 = 1000

Since there is no termination criteria, it is effectively an infinite-horizon task with a practical time limit.

Back to our environment, I only want to return a discount of 0.0 when the cube falls, which is a failure. Anything else is going to be, either the target orientation is reached in which case we return a discount of 1.0 and terminate.
