#!/usr/bin/env python3
"""CLI entrypoint for motion detection on warehouse pilot videos.

Handoff contract (output JSON per frame):
    {
        "frame_idx": int,
        "timestamp_sec": float,
        "motion_regions": [{"bbox": [x1, y1, x2, y2], "magnitude": float}, ...],
        "frame_motion_score": float,
        "event_type": "sustained" | "sudden_spike" | "none"
    }

Usage (single video):
    python -m cv_pipeline.motion_detection.run_motion_detection \
        --video data/pilot_videos/"Dock level, dragging cupboard.mp4" \
        --output motion_output.json \
        --fps 10 \
        --visualize

Usage (batch mode):
    python -m cv_pipeline.motion_detection.run_motion_detection \
        --batch \
        --output-dir cv_pipeline/motion_detection/outputs \
        --fps 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Ensure we can import motion_detector from the same package
sys.path.insert(0, str(Path(__file__).parent))
from motion_detector import MotionDetector, MotionResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motion detection for Damage-DNA warehouse videos."
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Path to input video file (required unless --batch is used).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("motion_output.json"),
        help="Output JSON file path (single-video mode).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all videos in data/pilot_videos/ in batch mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cv_pipeline/motion_detection/outputs"),
        help="Directory for batch-mode outputs (gitignored).",
    )
    parser.add_argument(
        "--method",
        choices=["farneback", "raft"],
        default="farneback",
        help="Optical flow method: farneback (default) or raft (requires torch+torchvision).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target processing FPS (frames will be skipped to approximate this rate).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Magnitude threshold for pixel-level motion detection.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum region area (pixels) to keep as a motion region.",
    )
    parser.add_argument(
        "--spike-std-multiplier",
        type=float,
        default=2.5,
        help="Std-dev multiplier for sudden-spike detection.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=15,
        help="Rolling window size for spike detection statistics.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save annotated debug video with motion boxes color-coded by event_type "
        "(yellow=sustained, red=sudden_spike).",
    )
    return parser.parse_args()


def process_video(
    video_path: Path,
    detector: MotionDetector,
    target_fps: float,
    visualize: bool,
) -> List[dict]:
    """Process a single video, return list of per-frame result dicts."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(round(src_fps / target_fps)))

    results: List[dict] = []
    prev_frame = None
    frame_idx = 0
    processed = 0

    # Video writer for visualization
    writer = None
    if visualize:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = video_path.with_suffix(".annotated.mp4")
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            target_fps,
            (width, height),
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        if prev_frame is None:
            prev_frame = frame
            frame_idx += 1
            continue

        result: MotionResult = detector.process_frame_pair(prev_frame, frame)

        timestamp = frame_idx / src_fps
        frame_data = {
            "frame_idx": processed,
            "timestamp_sec": round(timestamp, 3),
            "motion_regions": result.motion_regions,
            "frame_motion_score": round(result.frame_motion_score, 4),
            "event_type": result.event_type,
        }
        results.append(frame_data)

        if visualize:
            vis = draw_regions(frame, result)
            writer.write(vis)

        prev_frame = frame
        frame_idx += 1
        processed += 1

    cap.release()
    if writer:
        writer.release()
        print(f"  Saved annotated video: {out_path}")

    return results


def draw_regions(frame: np.ndarray, result: MotionResult) -> np.ndarray:
    """Draw motion regions on frame, color-coded by event_type."""
    vis = frame.copy()
    color_map = {
        "sustained": (0, 255, 255),   # Yellow (BGR)
        "sudden_spike": (0, 0, 255),  # Red
        "none": (255, 0, 0),          # Blue
    }
    for region in result.motion_regions:
        x1, y1, x2, y2 = region["bbox"]
        mag = region["magnitude"]
        color = color_map.get(result.event_type, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{mag:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    # Event type label
    cv2.putText(
        vis,
        f"Event: {result.event_type} | Score: {result.frame_motion_score:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return vis


def print_summary(video_name: str, results: List[dict]) -> None:
    """Print end-of-run summary for a video."""
    if not results:
        print(f"  {video_name}: No frames processed.")
        return

    sustained = sum(1 for r in results if r["event_type"] == "sustained")
    spikes = sum(1 for r in results if r["event_type"] == "sudden_spike")
    none = sum(1 for r in results if r["event_type"] == "none")
    avg_score = np.mean([r["frame_motion_score"] for r in results])

    print(f"\n  === {video_name} ===")
    print(f"  Frames processed: {len(results)}")
    print(f"  Avg motion score: {avg_score:.3f}")
    print(f"  Events: sustained={sustained}, sudden_spike={spikes}, none={none}")


def run_batch(
    pilot_dir: Path,
    output_dir: Path,
    target_fps: float,
    method: str,
    threshold: float,
    min_area: int,
    spike_std_mult: float,
    rolling_window: int,
    visualize: bool,
) -> None:
    """Process all videos in pilot_dir."""
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
    videos = sorted([p for p in pilot_dir.iterdir() if p.suffix in video_exts])

    if not videos:
        print(f"No video files found in {pilot_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    detector = MotionDetector(
        method=method,
        motion_threshold=threshold,
        min_region_area=min_area,
        spike_std_multiplier=spike_std_mult,
        rolling_window=rolling_window,
    )

    print(f"\nBatch processing {len(videos)} videos from {pilot_dir}...")
    all_summaries = {}

    for video_path in videos:
        print(f"\nProcessing: {video_path.name}")
        try:
            results = process_video(
                video_path, detector, target_fps, visualize
            )
            out_path = output_dir / f"{video_path.stem}_motion.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved JSON: {out_path}")
            print_summary(video_path.name, results)
            all_summaries[video_path.name] = {
                "sustained": sum(1 for r in results if r["event_type"] == "sustained"),
                "sudden_spike": sum(1 for r in results if r["event_type"] == "sudden_spike"),
                "avg_score": np.mean([r["frame_motion_score"] for r in results]),
            }
        except Exception as e:
            print(f"  ERROR: {e}")

    # Cross-video comparison for sanity check
    print("\n" + "=" * 60)
    print("SUSTAINED vs SPIKE COMPARISON (sanity check)")
    print("=" * 60)
    for name, stats in all_summaries.items():
        print(
            f"  {name:55s} | "
            f"sustained={stats['sustained']:3d}  "
            f"spikes={stats['sudden_spike']:3d}  "
            f"avg_score={stats['avg_score']:.3f}"
        )

    # Specific check: Throwing Mattresses should have more spikes than Dragging Cupboard
    mattress_key = "Throwing Mattresses.mp4"
    dragging_key = "Dock level, dragging cupboard.mp4"
    if mattress_key in all_summaries and dragging_key in all_summaries:
        mattress_spikes = all_summaries[mattress_key]["sudden_spike"]
        dragging_spikes = all_summaries[dragging_key]["sudden_spike"]
        print("\n  Key sanity check:")
        print(f"    Throwing Mattresses spikes: {mattress_spikes}")
        print(f"    Dragging Cupboard spikes:   {dragging_spikes}")
        if mattress_spikes > dragging_spikes:
            print("    [PASS] Throwing has more sudden spikes than dragging")
        else:
            print("    [FAIL] Expected more spikes in throwing video")


def main() -> int:
    args = parse_args()

    if not args.batch and not args.video:
        print("ERROR: --video is required unless --batch is used.", file=sys.stderr)
        return 1

    detector = MotionDetector(
        method=args.method,
        motion_threshold=args.threshold,
        min_region_area=args.min_area,
        spike_std_multiplier=args.spike_std_multiplier,
        rolling_window=args.rolling_window,
    )

    if args.batch:
        pilot_dir = Path("data/pilot_videos")
        run_batch(
            pilot_dir=pilot_dir,
            output_dir=args.output_dir,
            target_fps=args.fps,
            method=args.method,
            threshold=args.threshold,
            min_area=args.min_area,
            spike_std_mult=args.spike_std_multiplier,
            rolling_window=args.rolling_window,
            visualize=args.visualize,
        )
    else:
        results = process_video(
            args.video, detector, args.fps, args.visualize
        )
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.output}")
        print_summary(args.video.name, results)

    return 0


if __name__ == "__main__":
    sys.exit(main())