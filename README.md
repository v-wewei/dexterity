# Shadow Hand

This repository contains code related to my research involving the [Shadow Hand](https://www.shadowrobot.com/).

<img src="./assets/teaser.gif" width="40%"/>

## Todos

- [ ] Figure out joint torque and velocity limits. I submitted an [issue](https://github.com/shadow-robot/sr_core/issues/206).
- [x] Add torque sensors and fingertip sites.
- [ ] Implement inverse kinematics.

## Changelog


## Questions

1. Is it better to specify damping at the `<joint>` level or have an explicit velocity actuator with a damping gain?
2. How do I tune the `<visual>`, `<option>` and `<size>` properties?
3. How do I tune `<joint>` properties such as `<damping>`, `<armature>`, `<margin>` and `<frictionloss>`?
