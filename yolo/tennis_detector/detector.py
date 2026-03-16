"""Core detector implementation.

This module contains the TennisActionDetector class and all supporting helpers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .types import ActionCandidate, PoseFrame

logger = logging.getLogger(__name__)


class TennisActionDetector:
    """High-performance tennis action detector using YOLO pose."""

    UPPER_BODY_INDICES = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
    KEYPOINT_NAMES = {
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
    }

    def __init__(
        self,
        model_path: str = "yolo11m-pose.pt",
        device: str = "auto",
        trigger_threshold: float = 25.0,
        sparse_fps: float = 5.0,
        confidence_threshold: float = 0.5,
        action_window_sec: float = 2.0,
        min_keypoint_confidence: float = 0.3,
    ):
        if device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing TennisActionDetector on device: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.trigger_threshold = trigger_threshold
        self.sparse_fps = sparse_fps
        self.confidence_threshold = confidence_threshold
        self.action_window_sec = action_window_sec
        self.min_keypoint_confidence = min_keypoint_confidence

        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Trigger threshold: {trigger_threshold}, Sparse FPS: {sparse_fps}")

    def _get_pose_embedding(
        self,
        results: Any,
        frame_shape: Tuple[int, int],
        selection_mode: str = "largest",
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        if results is None or len(results) == 0:
            return None

        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return None

        keypoints_data = result.keypoints.data.cpu().numpy()  # (N,17,3)
        boxes_data = result.boxes.xyxy.cpu().numpy()  # (N,4)
        boxes_conf = result.boxes.conf.cpu().numpy()  # (N,)

        if len(keypoints_data) == 0:
            return None

        valid_mask = boxes_conf >= self.confidence_threshold
        if not np.any(valid_mask):
            return None

        keypoints_data = keypoints_data[valid_mask]
        boxes_data = boxes_data[valid_mask]
        boxes_conf = boxes_conf[valid_mask]

        if selection_mode == "largest":
            areas = (boxes_data[:, 2] - boxes_data[:, 0]) * (boxes_data[:, 3] - boxes_data[:, 1])
            selected_idx = int(np.argmax(areas))
        elif selection_mode == "center":
            frame_h, frame_w = frame_shape
            frame_center = np.array([frame_w / 2, frame_h / 2])
            box_centers = np.stack(
                [
                    (boxes_data[:, 0] + boxes_data[:, 2]) / 2,
                    (boxes_data[:, 1] + boxes_data[:, 3]) / 2,
                ],
                axis=1,
            )
            distances = np.linalg.norm(box_centers - frame_center, axis=1)
            selected_idx = int(np.argmin(distances))
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

        person_keypoints = keypoints_data[selected_idx]  # (17,3)
        person_bbox = boxes_data[selected_idx]  # (4,)

        upper_body_kpts = person_keypoints[self.UPPER_BODY_INDICES]  # (6,3)
        kpt_confidences = upper_body_kpts[:, 2]
        valid_kpts_mask = kpt_confidences >= self.min_keypoint_confidence

        if np.sum(valid_kpts_mask) < 3:
            return None

        upper_body_coords = upper_body_kpts[:, :2].copy()
        upper_body_coords[~valid_kpts_mask] = 0

        upper_body_confs = kpt_confidences.copy()
        upper_body_confs[~valid_kpts_mask] = 0

        avg_confidence = float(np.mean(kpt_confidences[valid_kpts_mask]))

        return upper_body_coords, person_bbox, avg_confidence, upper_body_confs

    def _calculate_pose_velocity(
        self,
        pose1: np.ndarray,
        pose2: np.ndarray,
        normalize: bool = True,
        bbox1: Optional[np.ndarray] = None,
        bbox2: Optional[np.ndarray] = None,
    ) -> float:
        if pose1 is None or pose2 is None:
            return 0.0

        valid_mask = (pose1.sum(axis=1) != 0) & (pose2.sum(axis=1) != 0)
        if np.sum(valid_mask) < 2:
            return 0.0

        valid_pose1 = pose1[valid_mask]
        valid_pose2 = pose2[valid_mask]

        displacement = valid_pose2 - valid_pose1

        if normalize and bbox1 is not None and bbox2 is not None:
            avg_bbox_size = np.mean(
                [
                    np.sqrt((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])),
                    np.sqrt((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])),
                ]
            )
            if avg_bbox_size > 0:
                displacement = displacement / avg_bbox_size * 100

        velocity = np.sqrt(np.sum(displacement**2))
        return float(velocity)

    def _calculate_stride(self, video_fps: float) -> int:
        return max(1, int(video_fps / self.sparse_fps))

    def sparse_scan(
        self,
        video_path: str,
        selection_mode: str = "largest",
        merge_window_sec: float = 0.5,
        verbose: bool = True,
    ) -> List[ActionCandidate]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / video_fps

        stride = self._calculate_stride(video_fps)

        logger.info(f"Video: {video_path.name}")
        logger.info(
            f"FPS: {video_fps:.2f}, Total frames: {total_frames}, Duration: {duration_sec:.2f}s"
        )
        logger.info(
            f"Sparse scanning with stride: {stride} (effective FPS: {video_fps/stride:.2f})"
        )

        pose_history: List[PoseFrame] = []
        velocity_history: List[Tuple[int, float]] = []

        prev_pose: Optional[np.ndarray] = None
        prev_bbox: Optional[np.ndarray] = None

        frame_idx = 0
        processed_count = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames:
                break

            results = self.model(frame, verbose=False)
            pose_data = self._get_pose_embedding(
                results,
                frame_shape=(frame_height, frame_width),
                selection_mode=selection_mode,
            )

            if pose_data is not None:
                keypoints, bbox, confidence, keypoint_confs = pose_data
                pose_history.append(
                    PoseFrame(
                        frame_idx=frame_idx,
                        keypoints=keypoints,
                        keypoint_confs=keypoint_confs,
                        bbox=bbox,
                        confidence=confidence,
                    )
                )

                if prev_pose is not None:
                    velocity = self._calculate_pose_velocity(
                        prev_pose,
                        keypoints,
                        normalize=True,
                        bbox1=prev_bbox,
                        bbox2=bbox,
                    )
                    velocity_history.append((frame_idx, velocity))

                    if verbose and velocity > self.trigger_threshold * 0.5:
                        timestamp = frame_idx / video_fps
                        logger.debug(
                            f"Frame {frame_idx} ({timestamp:.2f}s): velocity = {velocity:.2f}"
                        )

                prev_pose = keypoints
                prev_bbox = bbox
            else:
                prev_pose = None
                prev_bbox = None

            frame_idx += stride
            processed_count += 1

            if verbose and processed_count % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({processed_count} frames processed)")

        cap.release()

        logger.info(f"Sparse scan complete. Processed {processed_count} frames.")
        logger.info(f"Found {len(velocity_history)} velocity measurements.")

        candidates = self._extract_candidates(
            velocity_history,
            video_fps=video_fps,
            stride=stride,
            merge_window_sec=merge_window_sec,
        )

        logger.info(f"Identified {len(candidates)} action candidates.")
        return candidates

    def _extract_candidates(
        self,
        velocity_history: List[Tuple[int, float]],
        video_fps: float,
        stride: int,
        merge_window_sec: float,
    ) -> List[ActionCandidate]:
        """Extract sparse action candidates from velocity history.

        Previous implementation used threshold-crossing + time clustering, which
        tends to merge multiple strokes into one long window when the player keeps
        moving above the trigger threshold.

        New logic:
        - Smooth sparse velocity curve.
        - Detect local maxima (peaks) above trigger threshold.
        - Apply a minimum peak separation (uses merge_window_sec for backward-compat).
        - Convert each peak into a candidate window using valley splits + low-threshold
          boundaries, then enforce a minimum window length (action_window_sec).
        """

        if not velocity_history:
            return []

        frame_idxs = np.array([fi for fi, _ in velocity_history], dtype=int)
        v_raw = np.array([v for _, v in velocity_history], dtype=float)
        if frame_idxs.size < 3:
            return []

        # Smooth the sparse velocity curve (keep window small to preserve peaks)
        v = self._smooth_1d(v_raw, window=5)

        # Robust thresholds: low bounds each peak's window; high is a "definitely active" level.
        low_thresh, high_thresh = self._robust_hysteresis_thresholds(
            v, low_percentile=25.0, high_z=2.0
        )

        # Adaptive peak threshold: in some clips, the player is "always moving" and
        # raw threshold-crossing will over-trigger; in others, strokes are subtle.
        # Use a mid-level between robust low/high, but never below trigger_threshold.
        peak_thresh_adaptive = max(float(self.trigger_threshold), 0.5 * (float(low_thresh) + float(high_thresh)))

        # Step time of sparse velocities (seconds per sample)
        step_sec = float(stride) / float(video_fps) if video_fps > 0 else 0.0
        if step_sec <= 0:
            step_sec = 1.0 / max(1.0, self.sparse_fps)

        # Use merge_window_sec as "min peak separation" for backward compatibility.
        # Default to a more stroke-like separation to reduce candidate count.
        min_peak_distance_sec = max(0.80, float(merge_window_sec))
        min_peak_distance_steps = max(1, int(round(min_peak_distance_sec / step_sec)))

        # Require a small prominence to avoid counting jitter as a stroke peak,
        # while still being conservative about misses.
        def _find_raw_peaks(peak_thresh: float) -> List[int]:
            """Find peaks using a sliding-window maximum.

            Compared to immediate-neighbor maxima, this is more stable on sparse samples
            and reduces spurious peaks by requiring each peak to be the maximum within
            a local neighborhood.
            """

            peaks: List[int] = []
            # neighborhood ~ half of min peak distance
            k = max(1, min_peak_distance_steps // 2)

            i = 0
            while i < n:
                if v[i] < peak_thresh:
                    i += 1
                    continue

                lo = max(0, i - k)
                hi = min(n, i + k + 1)
                local_max = float(np.max(v[lo:hi]))
                if float(v[i]) < local_max:
                    i += 1
                    continue

                # Plateau handling: take center index of the plateau at local_max
                left = i
                right = i
                while left - 1 >= 0 and float(v[left - 1]) == local_max:
                    left -= 1
                while right + 1 < n and float(v[right + 1]) == local_max:
                    right += 1
                peak_i = (left + right) // 2

                # Guard against pure smoothing artifacts: require some raw support.
                raw_guard = float(v_raw[peak_i]) >= max(0.80 * float(self.trigger_threshold), 0.30 * peak_thresh)
                if raw_guard:
                    peaks.append(int(peak_i))

                i = right + 1

            return peaks

        # ---- peak detection (local maxima) ----
        n = int(v.size)

        raw_peaks = _find_raw_peaks(peak_thresh_adaptive)
        if not raw_peaks:
            raw_peaks = _find_raw_peaks(float(self.trigger_threshold))

        if not raw_peaks:
            logger.info("No peaks exceeded trigger threshold.")
            return []

        # ---- peak NMS by minimum temporal distance (keep strongest peaks) ----
        raw_peaks_sorted = sorted(raw_peaks, key=lambda idx: float(v[idx]), reverse=True)
        kept: List[int] = []
        for p in raw_peaks_sorted:
            if all(abs(p - q) > min_peak_distance_steps for q in kept):
                kept.append(p)
        kept = sorted(kept)

        # ---- build candidate windows around each peak ----
        hold_steps = max(1, int(round(0.25 / step_sec)))

        def _bound_left(p_idx: int, left_limit: int) -> int:
            below = 0
            j = p_idx
            while j > left_limit:
                if v[j] < low_thresh:
                    below += 1
                    if below >= hold_steps:
                        return min(p_idx, j + hold_steps)
                else:
                    below = 0
                j -= 1
            return left_limit

        def _bound_right(p_idx: int, right_limit: int) -> int:
            below = 0
            j = p_idx
            while j < right_limit:
                if v[j] < low_thresh:
                    below += 1
                    if below >= hold_steps:
                        return max(p_idx, j - hold_steps)
                else:
                    below = 0
                j += 1
            return right_limit

        starts = []
        ends = []
        for p in kept:
            starts.append(_bound_left(p, 0))
            ends.append(_bound_right(p, n - 1))

        # Split neighboring peaks by valley (prevents long continuous-motion segments)
        for a in range(len(kept) - 1):
            p1 = kept[a]
            p2 = kept[a + 1]
            if p2 <= p1:
                continue
            valley_rel = int(np.argmin(v[p1 : p2 + 1]))
            valley = p1 + valley_rel
            ends[a] = min(ends[a], valley)
            starts[a + 1] = max(starts[a + 1], valley + 1)

        min_window_frames = int(self.action_window_sec * video_fps)
        candidates: List[ActionCandidate] = []

        for idx, p in enumerate(kept):
            s_i = int(np.clip(starts[idx], 0, n - 1))
            e_i = int(np.clip(ends[idx], 0, n - 1))
            if e_i < s_i:
                s_i, e_i = e_i, s_i

            start_frame = int(frame_idxs[s_i])
            end_frame = int(frame_idxs[e_i])
            peak_frame = int(frame_idxs[p])
            peak_velocity = float(v_raw[p])

            # Enforce minimum window length for stable dense refinement
            if min_window_frames > 0 and (end_frame - start_frame) < min_window_frames:
                need = min_window_frames - (end_frame - start_frame)
                left = need // 2
                right = need - left
                start_frame = max(0, start_frame - left)
                end_frame = end_frame + right

            avg_v = float(np.mean(v_raw[s_i : e_i + 1])) if e_i >= s_i else float(peak_velocity)
            confidence = min(1.0, avg_v / (self.trigger_threshold * 2))

            candidates.append(
                ActionCandidate(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    peak_frame=peak_frame,
                    peak_velocity=float(peak_velocity),
                    timestamp_start=start_frame / video_fps,
                    timestamp_end=end_frame / video_fps,
                    confidence=float(confidence),
                )
            )

        return candidates

    def _create_candidate(
        self,
        trigger_group: List[Tuple[int, float]],
        video_fps: float,
        action_window_frames: int,
    ) -> ActionCandidate:
        peak_idx = max(range(len(trigger_group)), key=lambda i: trigger_group[i][1])
        peak_frame, peak_velocity = trigger_group[peak_idx]

        first_frame = trigger_group[0][0]
        last_frame = trigger_group[-1][0]

        start_frame = max(0, first_frame - action_window_frames // 2)
        end_frame = last_frame + action_window_frames // 2

        avg_velocity = float(np.mean([vel for _, vel in trigger_group]))
        confidence = min(1.0, avg_velocity / (self.trigger_threshold * 2))

        return ActionCandidate(
            start_frame=start_frame,
            end_frame=end_frame,
            peak_frame=peak_frame,
            peak_velocity=float(peak_velocity),
            timestamp_start=start_frame / video_fps,
            timestamp_end=end_frame / video_fps,
            confidence=float(confidence),
        )

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }
        cap.release()
        return info

    def _smooth_1d(self, x: np.ndarray, window: int = 7) -> np.ndarray:
        if x.size == 0:
            return x
        if window <= 1:
            return x.astype(float, copy=True)
        window = int(window)
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(x.astype(float, copy=False), kernel, mode="same")

    def _robust_hysteresis_thresholds(
        self,
        velocities: np.ndarray,
        low_percentile: float = 25.0,
        high_z: float = 2.5,
        min_separation_ratio: float = 1.2,
    ) -> Tuple[float, float]:
        v = velocities[np.isfinite(velocities)]
        v = v[v > 0]
        if v.size == 0:
            return 0.0, 0.0

        median = float(np.median(v))
        mad = float(np.median(np.abs(v - median)))
        robust_std = 1.4826 * mad

        low = float(np.percentile(v, low_percentile))
        high = float(median + high_z * robust_std)

        if high < low * min_separation_ratio:
            high = low * min_separation_ratio

        return low, high

    def _segment_actions_hysteresis(
        self,
        velocities: np.ndarray,
        high_thresh: float,
        low_thresh: float,
        end_hold_frames: int = 3,
    ) -> List[Tuple[int, int]]:
        segments: List[Tuple[int, int]] = []
        in_seg = False
        seg_start = 0
        last_above_low = -1
        low_run = 0

        for i, v in enumerate(velocities):
            if not in_seg:
                if v >= high_thresh:
                    in_seg = True
                    seg_start = i
                    last_above_low = i
                    low_run = 0
                continue

            if v >= low_thresh:
                last_above_low = i
                low_run = 0
            else:
                low_run += 1
                if low_run >= end_hold_frames:
                    seg_end = max(last_above_low, seg_start)
                    segments.append((seg_start, seg_end))
                    in_seg = False
                    low_run = 0
                    last_above_low = -1

        if in_seg:
            seg_end = max(last_above_low, seg_start)
            segments.append((seg_start, seg_end))

        return segments

    def _merge_segments_by_gap(
        self,
        segments: List[Tuple[int, int]],
        frame_indices_for_velocity: List[int],
        max_gap_frames: int,
    ) -> List[Tuple[int, int]]:
        if not segments:
            return []
        segments = sorted(segments, key=lambda x: x[0])
        merged: List[Tuple[int, int]] = [segments[0]]
        for s, e in segments[1:]:
            ps, pe = merged[-1]
            prev_end_frame = frame_indices_for_velocity[pe]
            next_start_frame = frame_indices_for_velocity[s]
            gap = next_start_frame - prev_end_frame
            if gap <= max_gap_frames:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    def _expand_segment_min_duration(
        self,
        seg: Tuple[int, int],
        peak_idx: int,
        frame_indices_for_velocity: List[int],
        video_fps: float,
        min_duration_sec: float,
    ) -> Tuple[int, int]:
        s, e = seg
        if min_duration_sec <= 0:
            return s, e

        duration_frames = frame_indices_for_velocity[e] - frame_indices_for_velocity[s]
        if duration_frames / video_fps >= min_duration_sec:
            return s, e

        target_frames = int(min_duration_sec * video_fps)
        need = max(0, target_frames - duration_frames)

        left = need // 2
        right = need - left

        new_s = max(0, s - left)
        new_e = min(len(frame_indices_for_velocity) - 1, e + right)

        new_s = min(new_s, peak_idx)
        new_e = max(new_e, peak_idx)

        return new_s, new_e

    def _fill_short_gaps_linear(
        self,
        xy: np.ndarray,
        conf: np.ndarray,
        max_gap: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        filled_xy = xy.copy()
        filled_conf = conf.copy()

        valid = np.isfinite(filled_xy[:, 0]) & np.isfinite(filled_xy[:, 1])
        idx = np.where(valid)[0]
        if idx.size < 2:
            return filled_xy, filled_conf

        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if gap <= 0 or gap > max_gap:
                continue
            xa, ya = filled_xy[a]
            xb, yb = filled_xy[b]
            for t in range(1, gap + 1):
                alpha = t / (gap + 1)
                filled_xy[a + t, 0] = xa + (xb - xa) * alpha
                filled_xy[a + t, 1] = ya + (yb - ya) * alpha
                filled_conf[a + t] = min(filled_conf[a], filled_conf[b])

        return filled_xy, filled_conf

    def _compute_joint_speeds(
        self,
        poses: List[Optional[PoseFrame]],
        joint_idx_in_upper: int,
        max_gap: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(poses)
        if n <= 1:
            return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

        xy = np.full((n, 2), np.nan, dtype=float)
        conf = np.zeros((n,), dtype=float)
        bboxes: List[Optional[np.ndarray]] = [None] * n

        for i, p in enumerate(poses):
            if p is None:
                continue
            bboxes[i] = p.bbox
            c = float(p.keypoint_confs[joint_idx_in_upper]) if p.keypoint_confs is not None else 0.0
            if c <= 0:
                continue
            pt = p.keypoints[joint_idx_in_upper]
            if pt is None or (float(pt[0]) == 0.0 and float(pt[1]) == 0.0):
                continue
            xy[i] = pt
            conf[i] = c

        xy, conf = self._fill_short_gaps_linear(xy, conf, max_gap=max_gap)

        speeds = np.zeros((n - 1,), dtype=float)
        step_conf = np.zeros((n - 1,), dtype=float)

        for i in range(1, n):
            if not (np.isfinite(xy[i - 1]).all() and np.isfinite(xy[i]).all()):
                continue
            delta = xy[i] - xy[i - 1]
            c = min(conf[i], conf[i - 1])
            if c <= 0:
                continue

            scale = 1.0
            bb1 = bboxes[i - 1]
            bb2 = bboxes[i]
            if bb1 is not None and bb2 is not None:
                s1 = np.sqrt(max(1.0, (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])))
                s2 = np.sqrt(max(1.0, (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])))
                scale = float(np.mean([s1, s2]))
                if scale <= 0:
                    scale = 1.0

            speeds[i - 1] = float(np.linalg.norm(delta) / scale * 100.0)
            step_conf[i - 1] = c

        return speeds, step_conf

    def _select_hit_frame(
        self,
        poses: List[Optional[PoseFrame]],
        frame_indices: List[int],
        velocities: np.ndarray,
        velocities_smooth: np.ndarray,
        start_vel_idx: int,
        end_vel_idx: int,
        peak_vel_idx: int,
        video_fps: float,
        candidate_half_window_sec: float = 0.15,
    ) -> Tuple[int, int, float]:
        
        if velocities.size == 0:
            peak_frame = frame_indices[min(max(peak_vel_idx + 1, 0), len(frame_indices) - 1)]
            return peak_frame, min(max(peak_vel_idx + 1, 0), len(frame_indices) - 1), 0.0

        lw_speed, lw_conf = self._compute_joint_speeds(poses, joint_idx_in_upper=4)
        rw_speed, rw_conf = self._compute_joint_speeds(poses, joint_idx_in_upper=5)
        le_speed, le_conf = self._compute_joint_speeds(poses, joint_idx_in_upper=2)
        re_speed, re_conf = self._compute_joint_speeds(poses, joint_idx_in_upper=3)

        seg_slice = slice(max(0, start_vel_idx), min(end_vel_idx + 1, velocities.size))
        lw_score = float(np.sum(lw_speed[seg_slice] * lw_conf[seg_slice]))
        rw_score = float(np.sum(rw_speed[seg_slice] * rw_conf[seg_slice]))
        dominant_is_right = rw_score >= lw_score

        wrist_speed = rw_speed if dominant_is_right else lw_speed
        wrist_step_conf = rw_conf if dominant_is_right else lw_conf
        elbow_speed = re_speed if dominant_is_right else le_speed
        elbow_step_conf = re_conf if dominant_is_right else le_conf

        half_w = max(1, int(candidate_half_window_sec * video_fps))
        cand_s = max(start_vel_idx, peak_vel_idx - half_w)
        cand_e = min(end_vel_idx, peak_vel_idx + half_w)
        if cand_s >= cand_e:
            cand_s, cand_e = start_vel_idx, end_vel_idx

        def _robust_norm(sig: np.ndarray, sl: slice) -> np.ndarray:
            x = sig[sl]
            x = x[np.isfinite(x)]
            if x.size == 0:
                return sig
            p95 = float(np.percentile(x, 95))
            denom = p95 if p95 > 1e-6 else float(np.max(x)) if float(np.max(x)) > 1e-6 else 1.0
            return sig / denom

        wrist_n = _robust_norm(wrist_speed, slice(cand_s, cand_e + 1))
        elbow_n = _robust_norm(elbow_speed, slice(cand_s, cand_e + 1))
        global_n = _robust_norm(velocities_smooth, slice(cand_s, cand_e + 1))

        best_idx = int(peak_vel_idx)
        best_score = -1.0

        for i in range(int(cand_s), int(cand_e) + 1):
            w_wrist = 1.0 * float(np.clip(wrist_step_conf[i], 0.0, 1.0))
            w_elbow = 0.7 * float(np.clip(elbow_step_conf[i], 0.0, 1.0))
            w_global = 0.35
            w_sum = w_wrist + w_elbow + w_global
            if w_sum <= 1e-6:
                continue
            score = (w_wrist * wrist_n[i] + w_elbow * elbow_n[i] + w_global * global_n[i]) / w_sum
            if score > best_score:
                best_score = float(score)
                best_idx = int(i)

        hit_pose_idx = min(max(best_idx + 1, 0), len(frame_indices) - 1)
        hit_frame = frame_indices[hit_pose_idx]
        return hit_frame, hit_pose_idx, float(best_score)

    def refine_action_clip(
        self,
        video_path: str,
        candidate: ActionCandidate,
        buffer_sec: float = 1.5,
        selection_mode: str = "largest",
        sigma_multiplier: float = 2.0,
        static_percentile: float = 25.0,
        verbose: bool = True,
    ) -> Optional[Dict[str, Any]]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buffer_frames = int(buffer_sec * video_fps)
        start_frame = max(0, candidate.start_frame - buffer_frames)
        end_frame = min(total_frames - 1, candidate.end_frame + buffer_frames)

        if verbose:
            logger.info(f"Dense refinement: frames {start_frame} - {end_frame}")
            logger.info(f"Buffer: ±{buffer_sec}s ({buffer_frames} frames)")

        poses: List[Optional[PoseFrame]] = []
        frame_indices: List[int] = []

        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                poses.append(None)
                frame_indices.append(frame_idx)
                continue

            results = self.model(frame, verbose=False)
            pose_data = self._get_pose_embedding(
                results,
                frame_shape=(frame_height, frame_width),
                selection_mode=selection_mode,
            )

            if pose_data is not None:
                keypoints, bbox, confidence, keypoint_confs = pose_data
                poses.append(
                    PoseFrame(
                        frame_idx=frame_idx,
                        keypoints=keypoints,
                        keypoint_confs=keypoint_confs,
                        bbox=bbox,
                        confidence=confidence,
                    )
                )
            else:
                poses.append(None)

            frame_indices.append(frame_idx)
            if verbose and (frame_idx - start_frame) % 30 == 0:
                progress = ((frame_idx - start_frame) / (end_frame - start_frame)) * 100
                logger.info(f"Dense processing: {progress:.1f}%")

        cap.release()

        velocities: List[float] = []
        velocity_frame_indices: List[int] = []

        for i in range(1, len(poses)):
            if poses[i] is not None and poses[i - 1] is not None:
                vel = self._calculate_pose_velocity(
                    poses[i - 1].keypoints,
                    poses[i].keypoints,
                    normalize=True,
                    bbox1=poses[i - 1].bbox,
                    bbox2=poses[i].bbox,
                )
            else:
                vel = 0.0
            velocities.append(float(vel))
            velocity_frame_indices.append(frame_indices[i])

        if len(velocities) == 0 or float(np.max(velocities)) == 0.0:
            logger.warning("No valid velocities calculated. Refinement failed.")
            return None

        velocities_array = np.array(velocities, dtype=float)

        non_zero_velocities = velocities_array[velocities_array > 0]
        if non_zero_velocities.size < 5:
            mu = float(np.mean(velocities_array))
            sigma = float(np.std(velocities_array))
        else:
            mu = float(np.mean(non_zero_velocities))
            sigma = float(np.std(non_zero_velocities))

        velocities_smooth = self._smooth_1d(velocities_array, window=7)

        static_thresh, action_thresh = self._robust_hysteresis_thresholds(
            velocities_smooth,
            low_percentile=static_percentile,
            high_z=max(1.0, sigma_multiplier),
        )

        if verbose:
            logger.info(f"Velocity stats (for reference): μ={mu:.2f}, σ={sigma:.2f}")
            logger.info(
                f"Robust thresholds: low(static)={static_thresh:.2f}, high(action)={action_thresh:.2f}"
            )

        candidate_start_local = max(0, candidate.start_frame - start_frame - 1)
        candidate_end_local = min(len(velocities) - 1, candidate.end_frame - start_frame)
        if candidate_start_local >= len(velocities):
            candidate_start_local = 0
        if candidate_end_local < 0:
            candidate_end_local = len(velocities) - 1

        window_velocities = velocities_smooth[candidate_start_local : candidate_end_local + 1]
        if len(window_velocities) == 0:
            peak_local_idx = int(np.argmax(velocities_smooth))
        else:
            peak_local_idx = int(candidate_start_local + np.argmax(window_velocities))

        peak_frame = velocity_frame_indices[peak_local_idx]
        if verbose:
            logger.info(
                f"Peak action frame: {peak_frame} (velocity={velocities_smooth[peak_local_idx]:.2f})"
            )

        end_hold_frames = max(2, int(0.10 * video_fps))
        segments = self._segment_actions_hysteresis(
            velocities_smooth,
            high_thresh=action_thresh,
            low_thresh=static_thresh,
            end_hold_frames=end_hold_frames,
        )

        merge_gap_sec = 0.20
        max_gap_frames = int(merge_gap_sec * video_fps)
        segments = self._merge_segments_by_gap(
            segments,
            frame_indices_for_velocity=velocity_frame_indices,
            max_gap_frames=max_gap_frames,
        )

        chosen: Optional[Tuple[int, int]] = None
        for s, e in segments:
            if s <= peak_local_idx <= e:
                chosen = (s, e)
                break
        if chosen is None and segments:
            chosen = max(segments, key=lambda se: float(np.max(velocities_smooth[se[0] : se[1] + 1])))
        if chosen is None:
            fallback_half = max(2, int(0.25 * video_fps))
            chosen = (
                max(0, peak_local_idx - fallback_half),
                min(len(velocities_smooth) - 1, peak_local_idx + fallback_half),
            )

        min_duration_sec = 0.8
        refined_start_local, refined_end_local = self._expand_segment_min_duration(
            chosen,
            peak_idx=peak_local_idx,
            frame_indices_for_velocity=velocity_frame_indices,
            video_fps=video_fps,
            min_duration_sec=min_duration_sec,
        )

        refined_start_frame = velocity_frame_indices[refined_start_local]
        refined_end_frame = velocity_frame_indices[refined_end_local]

        if verbose:
            logger.info(f"Refined window: frames {refined_start_frame} - {refined_end_frame}")
            duration = (refined_end_frame - refined_start_frame) / video_fps
            logger.info(f"Action duration: {duration:.3f}s")

        hit_frame, hit_pose_idx, hit_score = self._select_hit_frame(
            poses=poses,
            frame_indices=frame_indices,
            velocities=velocities_array,
            velocities_smooth=velocities_smooth,
            start_vel_idx=refined_start_local,
            end_vel_idx=refined_end_local,
            peak_vel_idx=peak_local_idx,
            video_fps=video_fps,
            candidate_half_window_sec=0.15,
        )

        if not np.isfinite(hit_score) or hit_score <= 0:
            hit_frame = peak_frame
            hit_pose_idx = peak_local_idx + 1
            hit_score = 0.0

        if verbose:
            logger.info(f"Hit/Contact frame: {hit_frame} (score={hit_score:.3f})")
            hit_timestamp = hit_frame / video_fps
            logger.info(f"Hit timestamp: {hit_timestamp:.3f}s")

        result = {
            "start_frame": refined_start_frame,
            "end_frame": refined_end_frame,
            "hit_frame": hit_frame,
            "peak_frame": peak_frame,
            "start_timestamp": refined_start_frame / video_fps,
            "end_timestamp": refined_end_frame / video_fps,
            "hit_timestamp": hit_frame / video_fps,
            "peak_timestamp": peak_frame / video_fps,
            "duration_sec": (refined_end_frame - refined_start_frame) / video_fps,
            "pose_velocities": velocities_array,
            "velocity_frame_indices": velocity_frame_indices,
            "static_thresh": static_thresh,
            "action_thresh": action_thresh,
            "velocity_mean": mu,
            "velocity_std": sigma,
            "hit_score": hit_score,
            "all_poses": poses,
            "frame_indices": frame_indices,
            "video_fps": video_fps,
            "buffer_start_frame": start_frame,
            "buffer_end_frame": end_frame,
        }

        return result

    def _apply_temporal_nms(self, candidates: List[ActionCandidate], iou_threshold: float = 0.3) -> List[ActionCandidate]:
        if len(candidates) <= 1:
            return candidates

        sorted_candidates = sorted(candidates, key=lambda x: x.peak_velocity, reverse=True)

        keep: List[ActionCandidate] = []
        suppressed: set[int] = set()

        for i, candidate_i in enumerate(sorted_candidates):
            if i in suppressed:
                continue

            keep.append(candidate_i)

            for j in range(i + 1, len(sorted_candidates)):
                if j in suppressed:
                    continue

                candidate_j = sorted_candidates[j]

                inter_start = max(candidate_i.start_frame, candidate_j.start_frame)
                inter_end = min(candidate_i.end_frame, candidate_j.end_frame)
                inter = max(0, inter_end - inter_start)

                union_start = min(candidate_i.start_frame, candidate_j.start_frame)
                union_end = max(candidate_i.end_frame, candidate_j.end_frame)
                union = max(1, union_end - union_start)

                iou = inter / union

                if iou > iou_threshold:
                    suppressed.add(j)

        keep = sorted(keep, key=lambda x: x.start_frame)
        logger.info(
            f"Temporal NMS: {len(candidates)} -> {len(keep)} candidates (suppressed {len(suppressed)})"
        )
        return keep

    def _apply_hit_timestamp_nms(self, refined_results: List[Dict[str, Any]], min_time_gap: float = 1.0) -> List[Dict[str, Any]]:
        if len(refined_results) <= 1:
            return refined_results

        sorted_results = sorted(refined_results, key=lambda x: x.get("hit_score", 0), reverse=True)

        keep: List[Dict[str, Any]] = []
        suppressed_indices: set[int] = set()

        for i, result_i in enumerate(sorted_results):
            if i in suppressed_indices:
                continue

            keep.append(result_i)
            hit_time_i = result_i["hit_timestamp"]

            for j in range(i + 1, len(sorted_results)):
                if j in suppressed_indices:
                    continue

                result_j = sorted_results[j]
                hit_time_j = result_j["hit_timestamp"]

                if abs(hit_time_j - hit_time_i) < min_time_gap:
                    suppressed_indices.add(j)

        keep = sorted(keep, key=lambda x: x["hit_timestamp"])
        logger.info(
            f"Post-refinement NMS: {len(refined_results)} -> {len(keep)} actions (suppressed {len(suppressed_indices)} duplicates)"
        )
        return keep

    def _get_full_pose_embedding(
        self,
        results: Any,
        frame_shape: Tuple[int, int],
        selection_mode: str = "largest",
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        if results is None or len(results) == 0:
            return None

        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return None

        keypoints_data = result.keypoints.data.cpu().numpy()
        boxes_data = result.boxes.xyxy.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()

        if len(keypoints_data) == 0:
            return None

        valid_mask = boxes_conf >= self.confidence_threshold
        if not np.any(valid_mask):
            return None

        keypoints_data = keypoints_data[valid_mask]
        boxes_data = boxes_data[valid_mask]

        if selection_mode == "largest":
            areas = (boxes_data[:, 2] - boxes_data[:, 0]) * (boxes_data[:, 3] - boxes_data[:, 1])
            selected_idx = int(np.argmax(areas))
        else:
            frame_h, frame_w = frame_shape
            frame_center = np.array([frame_w / 2, frame_h / 2])
            box_centers = np.stack(
                [
                    (boxes_data[:, 0] + boxes_data[:, 2]) / 2,
                    (boxes_data[:, 1] + boxes_data[:, 3]) / 2,
                ],
                axis=1,
            )
            distances = np.linalg.norm(box_centers - frame_center, axis=1)
            selected_idx = int(np.argmin(distances))

        person_keypoints = keypoints_data[selected_idx]  # (17,3)
        person_bbox = boxes_data[selected_idx]

        upper_body_kpts = person_keypoints[self.UPPER_BODY_INDICES]
        kpt_confidences = upper_body_kpts[:, 2]
        valid_kpts_mask = kpt_confidences >= self.min_keypoint_confidence

        if np.sum(valid_kpts_mask) < 3:
            return None

        upper_body_coords = upper_body_kpts[:, :2].copy()
        upper_body_coords[~valid_kpts_mask] = 0
        avg_confidence = float(np.mean(kpt_confidences[valid_kpts_mask]))

        return upper_body_coords, person_keypoints, person_bbox, avg_confidence

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        draw_bbox: bool = True,
        keypoint_color: Tuple[int, int, int] = (0, 255, 0),
        skeleton_color: Tuple[int, int, int] = (255, 255, 0),
        bbox_color: Tuple[int, int, int] = (0, 255, 255),
        keypoint_radius: int = 5,
        line_thickness: int = 2,
        label: str = "",
    ) -> np.ndarray:
        frame = frame.copy()

        skeleton_connections = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

        colors = {
            "face": (255, 200, 100),
            "upper_body": (0, 255, 255),
            "torso": (0, 255, 0),
            "lower_body": (255, 0, 255),
        }

        connection_colors = {
            (0, 1): "face",
            (0, 2): "face",
            (1, 3): "face",
            (2, 4): "face",
            (5, 6): "upper_body",
            (5, 7): "upper_body",
            (7, 9): "upper_body",
            (6, 8): "upper_body",
            (8, 10): "upper_body",
            (5, 11): "torso",
            (6, 12): "torso",
            (11, 12): "torso",
            (11, 13): "lower_body",
            (13, 15): "lower_body",
            (12, 14): "lower_body",
            (14, 16): "lower_body",
        }

        for (i, j) in skeleton_connections:
            if keypoints[i, 2] > self.min_keypoint_confidence and keypoints[j, 2] > self.min_keypoint_confidence:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                color = colors.get(connection_colors.get((i, j), "upper_body"), skeleton_color)
                cv2.line(frame, pt1, pt2, color, line_thickness)

        for i, kpt in enumerate(keypoints):
            if kpt[2] > self.min_keypoint_confidence:
                pt = (int(kpt[0]), int(kpt[1]))
                if i in self.UPPER_BODY_INDICES:
                    cv2.circle(frame, pt, keypoint_radius + 2, (0, 0, 255), -1)
                cv2.circle(frame, pt, keypoint_radius, keypoint_color, -1)

        if draw_bbox and bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            if label:
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0] + 10, y1),
                    bbox_color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

        return frame

    def extract_clip(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        output_path: str,
        padding_frames: int = 0,
    ) -> str:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = max(0, start_frame - padding_frames)
        end_frame = min(total_frames - 1, end_frame + padding_frames)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(end_frame - start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        logger.info(f"Extracted clip: {output_path}")
        return output_path

    def extract_keyframe(self, video_path: str, frame_idx: int, output_path: str) -> str:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(output_path, frame)
            logger.info(f"Extracted keyframe: {output_path}")
            return output_path
        raise RuntimeError(f"Failed to extract frame {frame_idx}")

    def extract_keyframe_with_skeleton(
        self,
        video_path: str,
        frame_idx: int,
        output_path: str,
        selection_mode: str = "largest",
        label: str = "",
    ) -> str:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            cap.release()
            raise RuntimeError(f"Failed to extract frame {frame_idx}")

        height, width = frame.shape[:2]
        cap.release()

        results = self.model(frame, verbose=False)
        pose_data = self._get_full_pose_embedding(
            results,
            frame_shape=(height, width),
            selection_mode=selection_mode,
        )

        if pose_data is not None:
            _, full_keypoints, bbox, _ = pose_data
            frame = self._draw_skeleton(frame, full_keypoints, bbox, draw_bbox=True, label=label)

        info_text = f"Frame: {frame_idx}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imwrite(output_path, frame)
        logger.info(f"Extracted keyframe with skeleton: {output_path}")
        return output_path

    def save_results(
        self,
        video_path: str,
        refined_results: List[Dict[str, Any]],
        output_dir: str,
        draw_skeleton: bool = True,
        export_clips: bool = True,
        export_json: bool = True,
        clip_padding_frames: int = 5,
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_name = Path(video_path).stem
        video_info = self.get_video_info(video_path)

        saved_files: Dict[str, Any] = {
            "clips": [],
            "keyframes": [],
            "json": None,
            "summary": {
                "video_name": video_name,
                "video_path": str(video_path),
                "total_actions": len(refined_results),
                "video_fps": video_info["fps"],
                "video_duration": video_info["duration_sec"],
                "processed_at": datetime.now().isoformat(),
            },
        }

        clip_buffer_sec = 1.5
        total_frames = video_info["total_frames"]
        fps = video_info["fps"]

        for i, result in enumerate(refined_results):
            action_id = result.get("action_id", i + 1)
            prefix = f"{video_name}_action_{action_id:03d}"

            if export_clips:
                hit_frame = int(result["hit_frame"])
                clip_start = max(0, hit_frame - int(clip_buffer_sec * fps))
                clip_end = min(total_frames - 1, hit_frame + int(clip_buffer_sec * fps))

                clip_path = str(output_dir / f"{prefix}.mp4")
                self.extract_clip(video_path, clip_start, clip_end, clip_path, padding_frames=clip_padding_frames)
                result["clip_path"] = clip_path
                saved_files["clips"].append(clip_path)

            if draw_skeleton:
                hit_path = str(output_dir / f"{prefix}_hit.jpg")
                start_path = str(output_dir / f"{prefix}_start.jpg")
                end_path = str(output_dir / f"{prefix}_end.jpg")

                self.extract_keyframe_with_skeleton(video_path, int(result["hit_frame"]), hit_path, label="HIT")
                self.extract_keyframe_with_skeleton(video_path, int(result["start_frame"]), start_path, label="START")
                self.extract_keyframe_with_skeleton(video_path, int(result["end_frame"]), end_path, label="END")

                result["hit_frame_path"] = hit_path
                result["start_frame_path"] = start_path
                result["end_frame_path"] = end_path

                saved_files["keyframes"].extend([hit_path, start_path, end_path])
            else:
                hit_path = str(output_dir / f"{prefix}_hit.jpg")
                self.extract_keyframe(video_path, int(result["hit_frame"]), hit_path)
                result["hit_frame_path"] = hit_path
                saved_files["keyframes"].append(hit_path)

        if export_json:
            json_path = str(output_dir / f"{video_name}_actions.json")

            json_data: Dict[str, Any] = {"summary": saved_files["summary"], "actions": []}
            for result in refined_results:
                action_data = {
                    k: v
                    for k, v in result.items()
                    if k
                    not in {
                        "pose_velocities",
                        "all_poses",
                        "frame_indices",
                        "velocity_frame_indices",
                    }
                }
                json_data["actions"].append(action_data)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            saved_files["json"] = json_path
            logger.info(f"Saved JSON metadata: {json_path}")

        logger.info("\n" + "=" * 60)
        logger.info("SAVE RESULTS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Video clips saved: {len(saved_files['clips'])}")
        logger.info(f"Keyframes saved: {len(saved_files['keyframes'])}")
        if saved_files["json"]:
            logger.info(f"JSON metadata: {saved_files['json']}")

        return saved_files

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        selection_mode: str = "largest",
        apply_nms: bool = True,
        nms_iou_threshold: float = 0.3,
        draw_skeleton: bool = True,
        export_clips: bool = True,
        export_json: bool = True,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        '''
        处理指定的视频
        
        :param self: Description
        :param video_path: Description
        :type video_path: str
        :param output_dir: Description
        :type output_dir: str
        :param selection_mode: Description
        :type selection_mode: str
        :param apply_nms: Description
        :type apply_nms: bool
        :param nms_iou_threshold: Description
        :type nms_iou_threshold: float
        :param draw_skeleton: Description
        :type draw_skeleton: bool
        :param export_clips: Description
        :type export_clips: bool
        :param export_json: Description
        :type export_json: bool
        :param verbose: Description
        :type verbose: bool
        :return: Description
        :rtype: List[Dict[str, Any]]
        '''
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_name = Path(video_path).stem
        video_info = self.get_video_info(video_path)

        logger.info("=" * 60)
        logger.info("TENNIS ACTION DETECTOR - FULL PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Video: {video_path}")
        logger.info(f"Duration: {video_info['duration_sec']:.2f}s @ {video_info['fps']:.2f} FPS")
        logger.info(f"Resolution: {video_info['width']}x{video_info['height']}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 60)

        logger.info("\n[PHASE 1] SPARSE SCANNING")
        logger.info("-" * 40)

        candidates = self.sparse_scan(video_path, selection_mode=selection_mode, verbose=verbose)
        if not candidates:
            logger.info("No action candidates found. Exiting.")
            return []

        logger.info(f"Found {len(candidates)} initial candidates")

        if apply_nms and len(candidates) > 1:
            logger.info("\n[PHASE 2] TEMPORAL NMS")
            logger.info("-" * 40)
            candidates = self._apply_temporal_nms(candidates, iou_threshold=nms_iou_threshold)

        logger.info("\n[PHASE 3] DENSE REFINEMENT")
        logger.info("-" * 40)

        refined_results: List[Dict[str, Any]] = []

        for i, candidate in enumerate(candidates):
            logger.info(f"\n--- Refining Action {i+1}/{len(candidates)} ---")
            logger.info(f"Candidate window: {candidate.timestamp_start:.2f}s - {candidate.timestamp_end:.2f}s")

            refined = self.refine_action_clip(video_path, candidate, selection_mode=selection_mode, verbose=verbose)
            if refined is None:
                logger.warning(f"Refinement failed for candidate {i+1}")
                continue

            refined["action_id"] = len(refined_results) + 1
            refined["confidence"] = candidate.confidence
            refined_results.append(refined)

        if not refined_results:
            logger.info("No refined actions found. Exiting.")
            return []

        if len(refined_results) > 1:
            refined_results = self._apply_hit_timestamp_nms(refined_results, min_time_gap=1.0)

        logger.info("\n[PHASE 4] SAVING RESULTS")
        logger.info("-" * 40)

        self.save_results(
            video_path,
            refined_results,
            str(output_dir),
            draw_skeleton=draw_skeleton,
            export_clips=export_clips,
            export_json=export_json,
        )

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total actions detected: {len(refined_results)}")
        logger.info(f"\nOutputs saved to: {output_dir}")

        return refined_results
