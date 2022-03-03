"""Shadow hand actuation."""

import dataclasses
import enum
from typing import Dict, Tuple

from shadow_hand.models.hands import shadow_hand_e_constants as consts


class Actuation(enum.Enum):
    """Available actuation methods for the hand."""

    POSITION = enum.auto()
    """In this mode, position actuators are created for the joints and tendons of the
    hand. Incoming control commands are interepreted as desired joint positions and the
    underlying position controller chooses the right force for each actuator to reach
    these positions.
    """

    EFFORT = enum.auto()


@dataclasses.dataclass(frozen=True)
class ActuatorParams:
    """PD parameters for the position actuators of the hand."""

    kp: float
    damping: float = 1.0


# Actuation limits for the hand.
ACTUATION_LIMITS: Dict[Actuation, Dict[consts.Actuators, Tuple[float, float]]] = {
    # For position control, actuation limits are joint limits.
    Actuation.POSITION: consts.ACTUATOR_CTRLRANGE,
}


_WR_DAMPING = 1.0
_FJ4_DAMPING = 0.1
_FJ3_DAMPING = 0.1
_FJ2_DAMPING = 0.1
_FJ1_DAMPING = 0.1
_TH_DAMPING = 0.2

_WR_GAIN = 20.0
_TH_GAIN = 3.0
_FJ3_GAIN = 2.0
_FJ2_GAIN = 2.0
_FJ1_GAIN = 0.8
_FJ4_GAIN = 2.0

ACTUATOR_PARAMS: Dict[Actuation, Dict[consts.Actuators, ActuatorParams]] = {
    Actuation.POSITION: {
        # Wrist.
        consts.Actuators.A_WRJ1: ActuatorParams(kp=_WR_GAIN, damping=_WR_DAMPING),
        consts.Actuators.A_WRJ0: ActuatorParams(kp=_WR_GAIN, damping=_WR_DAMPING),
        # First finger.
        consts.Actuators.A_FFJ3: ActuatorParams(kp=_FJ3_GAIN, damping=_FJ3_DAMPING),
        consts.Actuators.A_FFJ2: ActuatorParams(kp=_FJ2_GAIN, damping=_FJ2_DAMPING),
        consts.Actuators.A_FFJ1: ActuatorParams(kp=_FJ1_GAIN, damping=_FJ1_DAMPING),
        # Middle finger.
        consts.Actuators.A_MFJ3: ActuatorParams(kp=_FJ3_GAIN, damping=_FJ3_DAMPING),
        consts.Actuators.A_MFJ2: ActuatorParams(kp=_FJ2_GAIN, damping=_FJ2_DAMPING),
        consts.Actuators.A_MFJ1: ActuatorParams(kp=_FJ1_GAIN, damping=_FJ1_DAMPING),
        # Ring finger.
        consts.Actuators.A_RFJ3: ActuatorParams(kp=_FJ3_GAIN, damping=_FJ3_DAMPING),
        consts.Actuators.A_RFJ2: ActuatorParams(kp=_FJ2_GAIN, damping=_FJ2_DAMPING),
        consts.Actuators.A_RFJ1: ActuatorParams(kp=_FJ1_GAIN, damping=_FJ1_DAMPING),
        # Little finger.
        consts.Actuators.A_LFJ4: ActuatorParams(kp=_FJ4_GAIN, damping=_FJ4_DAMPING),
        consts.Actuators.A_LFJ3: ActuatorParams(kp=_FJ3_GAIN, damping=_FJ3_DAMPING),
        consts.Actuators.A_LFJ2: ActuatorParams(kp=_FJ2_GAIN, damping=_FJ2_DAMPING),
        consts.Actuators.A_LFJ1: ActuatorParams(kp=_FJ1_GAIN, damping=_FJ1_DAMPING),
        # Thumb.
        consts.Actuators.A_THJ4: ActuatorParams(kp=_TH_GAIN, damping=_TH_DAMPING),
        consts.Actuators.A_THJ3: ActuatorParams(kp=_TH_GAIN, damping=_TH_DAMPING),
        consts.Actuators.A_THJ2: ActuatorParams(kp=_TH_GAIN, damping=_TH_DAMPING),
        consts.Actuators.A_THJ1: ActuatorParams(kp=_TH_GAIN, damping=_TH_DAMPING),
        consts.Actuators.A_THJ0: ActuatorParams(kp=_TH_GAIN, damping=_TH_DAMPING),
    },
}
