"""CLI for the Detection & Tracking stage. Reads a video, writes tracking JSON.

Usage:
    python -m cv_pipeline.detection_tracking.run_detection_tracking \
        --video path/to/video.mp4 --output tracks.json --fps 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from cv_pipeline.detection_tracking.detection_tracking import Detector, Tracker, tracks_to_json


def main() -> int:
    p = argparse.ArgumentParser(description="Detection & Tracking stage.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("tracks.json"))
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    det, tracker = Detector(), Tracker()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Cannot open {args.video}", file=sys.stderr)
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, int(round(src_fps / args.fps)))
    out_frames, idx, out = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip != 0:
            idx += 1
            continue
        if args.max_frames and out >= args.max_frames:
            break
        detections = det.detect(frame)
        tracks = tracker.update(detections)
        out_frames.append(tracks_to_json(tracks, out, out / args.fps))
        idx += 1
        out += 1
    cap.release()
    args.output.write_text(json.dumps(out_frames, indent=2))
    print(f"{len(out_frames)} frames, {max((t['track_id'] for f in out_frames for t in f['tracks']), default=-1)+1} tracks -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())