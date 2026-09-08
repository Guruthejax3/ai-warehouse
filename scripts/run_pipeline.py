#!/usr/bin/env python3
"""Run one warehouse clip end-to-end through the full ReplayTwin pipeline.

Chain: motion detection -> segmentation/tracking -> trajectory -> behaviour
matching (DTW) -> physics cross-check -> risk scoring.

    usage: python scripts/run_pipeline.py "data/clips/Throwing Mattresses.mp4"

The motion detector (Farneback) lives in cv-pipeline/motion_detection; it is
imported here via sys.path so the pipeline package can consume it without
moving the upstream code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "cv-pipeline" / "motion_detection"))
sys.path.insert(0, str(ROOT))

from motion_detector import MotionDetector  # noqa: E402  (upstream, on sys.path)
from pipeline.config import load_config  # noqa: E402
from pipeline.matching.dtw_matcher import DTWMatcher  # noqa: E402
from pipeline.perception.pose import PoseEstimator  # noqa: E402
from pipeline.perception.segmentation import SegmentationTracker  # noqa: E402
from pipeline.physics.kinematics import KinematicsAnalyzer  # noqa: E402
from pipeline.risk.scoring import RiskScorer  # noqa: E402
from pipeline.trajectory.builder import TrajectoryBuilder  # noqa: E402


def run_clip(video_path: Path, max_frames: int = 0) -> int:
    cfg = load_config()
    fps = float(cfg["pipeline"]["target_fps"])

    motion = MotionDetector(method="farneback")
    seg = SegmentationTracker(
        target_fps=fps,
        merge_kernel_px=int(cfg["segmentation"].get("merge_kernel_px", 6)),
        max_new_tracks=int(cfg["segmentation"].get("max_new_tracks", 12)),
    )
    pose = PoseEstimator(enabled=bool(cfg["pose_estimation"].get("enabled", True)))
    builder = TrajectoryBuilder(
        smoother=cfg["trajectory"].get("smoother", "moving_average"),
        ma_window=int(cfg["trajectory"].get("ma_window", 5)),
        min_track_length=int(cfg["trajectory"].get("min_track_length", 10)),
        min_speed_mps=float(cfg["trajectory"].get("min_speed_mps", 0.05)),
        pixel_scale=float(cfg["physics"].get("pixel_scale", 0.01)),
        max_gap=int(cfg["trajectory"].get("max_gap_frames", 10)),
    )
    matcher = DTWMatcher(
        exemplar_path=cfg["matching"].get("exemplar_path", "data/exemplars/behaviors.json"),
        algorithm=cfg["matching"].get("algorithm", "fastdtw"),
    )
    physics = KinematicsAnalyzer(
        pixel_scale=float(cfg["physics"].get("pixel_scale", 0.01)),
        gravity=float(cfg["physics"].get("gravity", 9.81)),
    )
    risk = RiskScorer(
        weights=cfg["risk"].get("weights"),
        fragility_classes=cfg["risk"].get("fragility_classes"),
        mandatory_before_alert=bool(cfg["physics"].get("mandatory_before_alert", True)),
        evidence_trim_sec=int(cfg["risk"].get("evidence_trim_sec", 5)),
        target_fps=fps,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}", file=sys.stderr)
        return 1

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))

    prev = None
    frame_idx = 0
    out_idx = 0
    n_events = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if out_idx % step != 0:
            out_idx += 1
            continue
        out_idx += 1
        ts = out_idx / fps
        if max_frames and frame_idx >= max_frames:
            break
        frame_idx += 1

        if prev is None:
            prev = frame
            continue

        res = motion.process_frame_pair(prev, frame)
        tracked = seg.process(frame, res.motion_regions, frame_idx, ts)
        builder.update(tracked, frame_idx, ts)
        prev = frame

        # Close a track and evaluate it whenever it ends; simplest for demo:
        # evaluate only at the end (see finalize below).
        _ = pose

    cap.release()

    # Finalize all tracks and score every sealed trajectory
    trajectories = builder.finalize()
    print(f"Processed {frame_idx} frames; finalized {len(trajectories)} tracks")

    events = []
    for traj in trajectories:
        behavior = matcher.match_trajectory(traj)
        ph = physics.analyze(traj)
        ok, reasons = physics.cross_check(behavior.behavior_class, ph)
        ev = risk.score_event(
            source_video=str(video_path),
            frame_idx=int(traj.points[-1].frame_idx),
            timestamp_sec=float(traj.points[-1].timestamp_sec),
            behavior=behavior,
            physics=ph,
            physics_ok=ok,
            physics_reasons=reasons,
        )
        events.append(ev)

    events.sort(key=lambda e: e.risk_score, reverse=True)
    for ev in events:
        flag = "ALERT" if ev.metadata["gated_by_physics"] is False else "gated"
        print(
            f"  [{ev.risk_level.upper():8s}] {ev.behavior_class:28s} "
            f"score={ev.risk_score:.2f} {flag:6s} track={ev.trajectory_id}"
        )
        if ev.risk_level in ("high", "critical"):
            print(f"      -> {ev.justification[:180]}")
        n_events += 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a warehouse clip")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N processed frames (0 = all)")
    args = parser.parse_args()
    return run_clip(args.video, max_frames=args.max_frames)


if __name__ == "__main__":
    sys.exit(main())
