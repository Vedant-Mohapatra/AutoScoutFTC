"""AutoScoutFTC -- automatic scouting of artifact-shooting robots from FTC match video."""

from .config import Config, DEFAULT, RobotParams, ShotParams
from .pipeline import MatchResult, RobotStats, analyse

__all__ = [
    "Config",
    "DEFAULT",
    "MatchResult",
    "RobotParams",
    "RobotStats",
    "ShotParams",
    "analyse",
]

__version__ = "0.1.0"
