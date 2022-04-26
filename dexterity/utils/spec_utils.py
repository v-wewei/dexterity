"""Utilities for working with dm_env specs."""

from typing import Sequence

import numpy as np
from dm_env import specs


# Reference: https://github.com/deepmind/dm_robotics/blob/main/py/agentflow/spec_utils.py
def merge_specs(spec_list: Sequence[specs.BoundedArray]) -> specs.BoundedArray:
    """Merges a list of `BoundedArray` specs into one."""

    # Check all specs are flat.
    for spec in spec_list:
        if len(spec.shape) > 1:
            raise ValueError("Not merging multi-dimensional spec: {}".format(spec))

    # Filter out no-op specs with no actuators.
    spec_list = [spec for spec in spec_list if spec.shape and spec.shape[0]]
    dtype = np.find_common_type([spec.dtype for spec in spec_list], [])

    num_actions = 0
    name = ""
    mins = np.array([], dtype=dtype)
    maxs = np.array([], dtype=dtype)

    for i, spec in enumerate(spec_list):
        num_actions += spec.shape[0]
        if name:
            name += "\t"
        name += spec.name or f"spec_{i}"
        mins = np.concatenate([mins, spec.minimum])
        maxs = np.concatenate([maxs, spec.maximum])

    return specs.BoundedArray(
        shape=(num_actions,), dtype=dtype, minimum=mins, maximum=maxs, name=name
    )
