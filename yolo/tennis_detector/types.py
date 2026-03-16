from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ActionCandidate:
    """Data class representing a candidate action window."""

    start_frame: int
    end_frame: int
    peak_frame: int
    peak_velocity: float
    timestamp_start: float  # seconds
    timestamp_end: float
    confidence: float


@dataclass
class RefinedAction:
    """Data class representing a fully refined action with all metadata."""

    action_id: int
    start_frame: int
    end_frame: int
    hit_frame: int
    peak_frame: int
    start_timestamp: float
    end_timestamp: float
    hit_timestamp: float
    peak_timestamp: float
    duration_sec: float
    hit_score: float
    static_thresh: float
    action_thresh: float
    confidence: float
    clip_path: Optional[str] = None
    hit_frame_path: Optional[str] = None
    start_frame_path: Optional[str] = None
    end_frame_path: Optional[str] = None
    hit_pose: Optional[np.ndarray] = None
    hit_bbox: Optional[np.ndarray] = None


@dataclass
class PoseFrame:
    """Pose data for a single frame."""

    frame_idx: int
    keypoints: np.ndarray  # (6,2) upper body: shoulders, elbows, wrists
    keypoint_confs: np.ndarray  # (6,) confidences for those upper body points
    bbox: np.ndarray  # (4,) [x1,y1,x2,y2]
    confidence: float
    full_keypoints: Optional[np.ndarray] = None  # (17,3) for drawing
