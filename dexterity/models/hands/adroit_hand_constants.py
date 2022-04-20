from pathlib import Path
from typing import Dict, Tuple

from dexterity import _SRC_ROOT
from dexterity.models.hands.shadow_hand_e_constants import Components

# Path to the shadow hand E series XML file.
ADROIT_HAND_E_XML: Path = (
    _SRC_ROOT
    / "models"
    / "vendor"
    / "adroit"
    / "adroit_hand_description"
    / "mjcf"
    / "adroit_hand.xml"
)

# Mapping from finger `Components` to its associated geoms.
FINGER_GEOM_MAPPING: Dict[Components, Tuple[str, ...]] = {
    Components.FF: ("V_ffproximal", "V_ffmiddle", "V_ffdistal"),
    Components.MF: ("V_mfproximal", "V_mfmiddle", "V_mfdistal"),
    Components.RF: ("V_rfproximal", "V_rfmiddle", "V_rfdistal"),
    Components.LF: ("V_lfproximal", "V_lfmiddle", "V_lfdistal"),
    Components.TH: ("V_thproximal", "V_thmiddle", "V_thdistal"),
}
