"""
    python tennis_action_detector.py video.mp4 --output ./output
"""

from __future__ import annotations

import logging

from tennis_detector import ActionCandidate, PoseFrame, RefinedAction, TennisActionDetector
from tennis_detector.cli import main

__all__ = [
    "ActionCandidate",
    "PoseFrame",
    "RefinedAction",
    "TennisActionDetector",
    "main",
]


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(main())