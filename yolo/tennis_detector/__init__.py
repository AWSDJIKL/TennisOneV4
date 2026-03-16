"""Tennis action detection package.

This package contains the detector implementation and CLI entrypoints.

Backwards-compatible entry script remains at project root: tennis_action_detector.py
"""

from .types import ActionCandidate, RefinedAction, PoseFrame
from .detector import TennisActionDetector

__all__ = [
    "ActionCandidate",
    "RefinedAction",
    "PoseFrame",
    "TennisActionDetector",
]
