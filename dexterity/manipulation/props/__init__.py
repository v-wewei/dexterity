"""Composer entities corresponding to props.

A "prop" is typically a non-actuated entity representing an object in the world.
"""

from dexterity.manipulation.props.juggling_ball import JugglingBall
from dexterity.manipulation.props.openai_cube import OpenAICube
from dexterity.manipulation.props.target_sphere import TargetSphere

__all__ = [
    "OpenAICube",
    "TargetSphere",
    "JugglingBall",
]
