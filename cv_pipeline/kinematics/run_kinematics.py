"""CLI for the Kinematics stage. Reads tracking JSON, writes kinematics JSON.

Usage:
    python -m cv_pipeline.kinematics.run_kinematics \
        --tracks tracks.json --output kinematics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cv_pipeline.kinematics.kinematics import KinematicsAnalyzer


def main() -> int:
    p = argparse.ArgumentParser(description="Kinematics stage.")
    p.add_argument("--tracks", type=Path, required=True, help="tracking JSON from stage 2.")
    p.add_argument("--output", type=Path, default=Path("kinematics.json"))
    p.add_argument("--pixel-scale", type=float, default=0.01)
    args = p.parse_args()

    frames = json.loads(args.tracks.read_text())
    trajs = KinematicsAnalyzer.trajectories_from_tracking(frames)
    kin = KinematicsAnalyzer(pixel_scale=args.pixel_scale)
    out = []
    for tid, tr in trajs.items():
        out.append(kin.analyze(
            tid, tr["xs"], tr["ys"], tr["ts"], class_label=tr["class_label"]
        ).to_dict())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"{len(out)} tracks -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())