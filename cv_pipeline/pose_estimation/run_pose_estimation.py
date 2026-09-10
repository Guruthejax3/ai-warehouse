"""CLI for the Pose Estimation stage. Reads frames + person tracks, writes poses.

Usage:
    python -m cv_pipeline.pose_estimation.run_pose_estimation \
        --video path/to/video.mp4 --tracks tracks.json --output poses.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from cv_pipeline.pose_estimation.pose_estimation import PoseEstimator


def main() -> int:
    p = argparse.ArgumentParser(description="Pose Estimation stage.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True, help="tracking JSON from stage 2.")
    p.add_argument("--output", type=Path, default=Path("poses.json"))
    p.add_argument("--backend", choices=["box", "mediapipe"], default="box")
    args = p.parse_args()

    tracks_by_frame = json.loads(args.tracks.read_text())
    peg = PoseEstimator(backend=args.backend)
    cap = cv2.VideoCapture(str(args.video))
    out = []
    fidx = 0
    for frame_like in tracks_by_frame:
        ok, frame = cap.read()
        if not ok:
            break
        fidx = int(frame_like["frame_idx"])
        ts = float(frame_like.get("timestamp_sec", 0.0))
        for t in frame_like.get("tracks", []):
            if t.get("bbox") is None:
                continue
            res = peg.process((t["bbox"][0], t["bbox"][1], t["bbox"][2], t["bbox"][3]),
                              frame=frame)
            if res:
                d = res.to_dict()
                d["track_id"] = int(t["track_id"])
                d["frame_idx"] = fidx
                d["timestamp_sec"] = round(ts, 3)
                out.append(d)
    cap.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"{len(out)} pose samples -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())