from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

from .detector import TennisActionDetector


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tennis Action Detector - Extract keyframes and clips from tennis videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\nExamples:\n"
            "  # Basic usage - process video and save results\n"
            "  python tennis_action_detector.py video.mp4 --output ./output\n\n"
            "  # Use larger model for better accuracy\n"
            "  python tennis_action_detector.py video.mp4 --model yolo11m-pose.pt --output ./output\n\n"
            "  # Adjust sensitivity (lower = more sensitive)\n"
            "  python tennis_action_detector.py video.mp4 --threshold 30.0 --output ./output\n\n"
            "  # Sparse scan only (quick preview)\n"
            "  python tennis_action_detector.py video.mp4 --scan-only\n"
        ),
    )

    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./tennis_output",
        help="Output directory for clips and keyframes (default: ./tennis_output)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m-pose.pt",
        help="YOLO Pose model path (default: yolo11m-pose.pt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=25.0,
        help="Trigger threshold for pose velocity (default: 25.0)",
    )
    parser.add_argument(
        "--sparse-fps",
        type=float,
        default=5.0,
        help="Target FPS for sparse scanning (default: 5.0)",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="largest",
        choices=["largest", "center"],
        help="Player selection mode (default: largest)",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.3,
        help="Temporal NMS IoU threshold (default: 0.3)",
    )
    parser.add_argument("--no-skeleton", action="store_true", help="Don't draw skeleton on keyframes")
    parser.add_argument("--no-clips", action="store_true", help="Don't export video clips")
    parser.add_argument("--no-json", action="store_true", help="Don't export JSON metadata")
    parser.add_argument("--scan-only", action="store_true", help="Only run sparse scan (no refinement)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    print(f"\n{'='*60}")
    print("TENNIS ACTION DETECTOR")
    print(f"{'='*60}")

    detector = TennisActionDetector(
        model_path=args.model,
        trigger_threshold=args.threshold,
        sparse_fps=args.sparse_fps,
    )

    if args.scan_only:
        print(f"\nRunning sparse scan on: {args.video}")
        print(f"{'='*60}\n")

        candidates = detector.sparse_scan(args.video, selection_mode=args.selection, verbose=not args.quiet)

        print(f"\n{'='*60}")
        print("DETECTED ACTION CANDIDATES (Sparse Scan)")
        print(f"{'='*60}")

        for i, candidate in enumerate(candidates, 1):
            print(
                f"{i:02d}. {candidate.timestamp_start:.2f}s - {candidate.timestamp_end:.2f}s "
                f"(peak={candidate.peak_frame}, v={candidate.peak_velocity:.2f}, conf={candidate.confidence:.2f})"
            )

        print(f"\n{'='*60}")
        print(f"Total: {len(candidates)} candidates found")
        print("Run without --scan-only to refine and extract clips")
        print(f"{'='*60}\n")
        return 0

    results = detector.process_video(
        args.video,
        args.output,
        selection_mode=args.selection,
        apply_nms=True,
        nms_iou_threshold=args.nms_threshold,
        draw_skeleton=not args.no_skeleton,
        export_clips=not args.no_clips,
        export_json=not args.no_json,
        verbose=not args.quiet,
    )

    if results:
        print(f"\n{'='*60}")
        print(f"SUCCESS! Processed {len(results)} tennis actions")

        # Print brief per-action hit info (frame, timestamp, bbox center if available)
        for i, r in enumerate(results, 1):
            hit_frame = r.get("hit_frame")
            hit_ts = r.get("hit_timestamp")
            hit_bbox = r.get("hit_bbox")
            center_txt = ""
            if hit_bbox:
                try:
                    x1, y1, x2, y2 = hit_bbox
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    center_txt = f", bbox_center=({cx:.1f},{cy:.1f})"
                except Exception:
                    center_txt = ""

            print(f"  {i:02d}. hit_frame={hit_frame}, hit_time={hit_ts:.2f}s{center_txt}")

        print(f"Output saved to: {args.output}")
        print(f"{'='*60}\n")
        return 0

    print(f"\n{'='*60}")
    print("No actions detected")
    print(f"{'='*60}\n")
    return 1
