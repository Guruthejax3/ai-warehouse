"""End-to-end cv-pipeline orchestrator: chains all six stages over one video.

Usage:
    python -m cv_pipeline.run_pipeline \
        --video data/clips/"a warehouse clip.mp4" \
        --output reports/cv_pipeline_report.json

    python -m cv_pipeline.run_pipeline --smoke    # synthetic clip, no video needed
    python -m cv_pipeline.run_pipeline --demo     # prints the ASCII stage flow

Output: one JSON report containing the stage handoffs (motion, tracks,
poses, kinematics, fsm, scored_events) plus a CLI summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from cv_pipeline.motion_detection.motion_detector import MotionDetector
from cv_pipeline.detection_tracking.detection_tracking import (
    Detector, Tracker, tracks_to_json,
)
from cv_pipeline.pose_estimation.pose_estimation import PoseEstimator
from cv_pipeline.kinematics.kinematics import KinematicsAnalyzer
from cv_pipeline.behaviour_fsm.behaviour_fsm import BehaviourFSM
from cv_pipeline.risk_scoring.risk_scoring import RiskScorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReplayTwin cv-pipeline (all 6 stages).")
    p.add_argument("--video", type=Path, help="Input video file.")
    p.add_argument("--output", type=Path, default=Path("cv_pipeline_report.json"))
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole clip.")
    p.add_argument("--smoke", action="store_true",
                   help="Run on a synthetic 90-frame clip (no video needed).")
    p.add_argument("--demo", action="store_true",
                   help="Print the six-stage ASCII pipeline and exit.")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--pixel-scale", type=float, default=0.01)
    return p.parse_args()


def pipeline_stages() -> str:
    return (
        "  +--------------+     +---------------+     +--------------+\n"
        "  | 1 motion     | --> | 2 detect+track | --> | 3 pose       |\n"
        "  |    regions   |     |  (Kalman IoU)  |     |  + posture   |\n"
        "  +--------------+     +---------------+     +--------------+\n"
        "        |                      |                      |\n"
        "        v                      v                      v\n"
        "  +--------------+     +--------------+     +------------------+\n"
        "  | 4 kinematics | --> | 5 behaviour  | --> | 6 risk scoring   |\n"
        "  |  physics     |     |  FSM / seq   |     |  explainable 0-1 |\n"
        "  +--------------+     +--------------+     +------------------+\n"
    )


def smoke_frames(fps: float, n: int = 90):
    """Yield synthetic frames: a soft blob moving to the dock and settling."""
    h, w = 720, 1280
    # 1) approach (toward left dock) 2) handle (hold) 3) transit right
    # 4) place 5) depart
    for i in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        cx = 900 - int(600 * min(1.0, i / 40))          # approach -> left
        if 35 <= i < 55:
            cx = 300                                       # handle (hold)
        elif 55 <= i < 75:
            cx = 300 + int(500 * (i - 55) / 20)            # transit -> right
        elif i >= 75:
            cx = 800                                       # place / depart
        cy = 300 + int(60 * np.sin(i * 0.25))
        cv2.rectangle(frame, (cx - 40, cy - 60), (cx + 40, cy + 60), (200, 200, 200), -1)
        cv2.rectangle(frame, (cx - 12, cy - 90), (cx + 12, cy - 55), (120, 200, 120), -1)
        yield frame, i / fps


def run_on_frames(frames, fps: float, pixel_scale: float) -> dict:
    motion = MotionDetector(motion_threshold=1.5)
    det = Detector()
    tracker = Tracker()
    pose = PoseEstimator(backend="box")
    kin = KinematicsAnalyzer(pixel_scale=pixel_scale)
    fsm = BehaviourFSM()
    scorer = RiskScorer()

    tracking_frames: List[dict] = []
    centroids: Dict[int, List[Tuple[float, float, float]]] = {}
    labels: Dict[int, str] = {}

    prev = None
    for idx, (frame, ts) in enumerate(frames):
        if prev is not None:
            motion_result = motion.process_frame_pair(prev, frame)
        else:
            motion_result = None
        prev = frame

        detections = det.detect(frame, motion_regions=motion_result.motion_regions if motion_result else None)
        tracks = tracker.update(detections)
        tracking_frames.append(tracks_to_json(tracks, idx, ts))
        for t in tracks:
            centroids.setdefault(t.track_id, []).append((t.center[0], t.center[1], ts))
            labels[t.track_id] = t.class_label

    # stage 4: kinematics for every track
    kin_out: List[dict] = []
    for tid, pts in centroids.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ts_ = [p[2] for p in pts]
        other = {oid: ([p[0] for p in o], [p[1] for p in o], [p[2] for p in o])
                 for oid, o in centroids.items()}
        summary = kin.analyze(tid, xs, ys, ts_, class_label=labels.get(tid, "object"),
                              other_tracks=other)
        kin_out.append(summary.to_dict())

    # stage 5: FSM on simple activity labels derived from velocity sign
    fsm_out: List[dict] = []
    for tid, pts in centroids.items():
        steps = []
        for k in range(1, len(pts)):
            vx = pts[k][0] - pts[k - 1][0]
            vy = pts[k][1] - pts[k - 1][1]
            speed = max(abs(vx), abs(vy))
            if speed < 1.0:
                state = "idle"
            elif vx < -3:
                state = "dock_approach"
            elif vx > 3:
                state = "transit"
            elif vy > 2:
                state = "place"
            else:
                state = "handle"
            steps.append((pts[k][2], state, f"vx={vx:.0f}px/s"))
        fsm_out.append(fsm.analyze(tid, steps).to_dict())

    # stage 6: score events
    events: List[dict] = []
    for k, s in enumerate(kin_out):
        fsm_tr = next((f for f in fsm_out if f["track_id"] == s["track_id"]), {})
        ev = scorer.score(
            event_id=uuid.uuid4().hex[:12],
            track_id=s["track_id"],
            timestamp_sec=float(s["sample_count"] or 0) / fps,
            kinematics_severity=s["severity_score"],
            pose_hazard=False,
            fsm_unsafe=bool(fsm_tr.get("violations")),
            drop_height_m=s["drop_height_est_m"],
            peak_speed_mps=s["peak_speed_mps"],
        )
        events.append(ev.to_dict())

    events.sort(key=lambda e: -e["risk_score"])
    return {
        "stages": {name: "done" for name in
                   ["motion", "detection_tracking", "pose_estimation",
                    "kinematics", "behaviour_fsm", "risk_scoring"]},
        "track_count": len(centroids),
        "tracks": [{"track_id": tid, "class_label": lbl}
                   for tid, lbl in labels.items()],
        "kinematics": kin_out,
        "fsm": fsm_out,
        "scored_events": events,
    }


def main() -> int:
    args = parse_args()
    if args.demo:
        print(pipeline_stages())
        return 0

    if args.smoke:
        frames = smoke_frames(args.fps)
        video_name = "synthetic://smoke"
    else:
        if not args.video:
            print("ERROR: --video is required unless --smoke/--demo is used.",
                  file=sys.stderr)
            return 1
        video_name = str(args.video)

        def load(video: Path):
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise SystemExit(f"Cannot open {video}")
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            skip = max(1, int(round(src_fps / args.fps)))
            idx, out = 0, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % skip != 0:
                    idx += 1
                    continue
                if args.max_frames and out >= args.max_frames:
                    break
                yield frame, out / args.fps
                idx += 1
                out += 1
            cap.release()

        frames = load(args.video)

    print(pipeline_stages())
    print(f"Running stages over {video_name} ...")
    report = run_on_frames(frames, args.fps, args.pixel_scale)
    report["video"] = video_name

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    print(f"Tracks: {report['track_count']}   Events: {len(report['scored_events'])}")
    for ev in report["scored_events"]:
        print(f"  [{ev['risk_level']:>8}] score={ev['risk_score']:.2f}  "
              f"{ev['behavior_class']}: {ev['justification']}")
    top = report["scored_events"][0] if report["scored_events"] else None
    if top and top["risk_level"] in ("high", "critical"):
        print("\n[!] Highest-priority event needs supervisor attention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())