"""CLI for the Behaviour FSM stage. Reads kinematics JSON, writes FSM traces.

Usage:
    python -m cv_pipeline.behaviour_fsm.run_behaviour_fsm \
        --kinematics kinematics.json --output fsm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cv_pipeline.behaviour_fsm.behaviour_fsm import BehaviourFSM


def main() -> int:
    p = argparse.ArgumentParser(description="Behaviour FSM stage.")
    p.add_argument("--kinematics", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("fsm.json"))
    p.add_argument("--min-dwell", type=float, default=0.4)
    args = p.parse_args()

    kdata = json.loads(args.kinematics.read_text())
    fsm = BehaviourFSM(min_dwell_sec=args.min_dwell)
    out = []
    for k in kdata:
        tid = int(k["track_id"])
        # derive activity steps from kinematic severity buckets (demo mapping)
        steps = [(0.0, "idle", "start")]
        n = int(k.get("sample_count", 3))
        for i in range(1, n):
            t = i / 10.0
            if float(k.get("peak_speed_mps", 0)) > 0.8 and i > n * 0.6:
                state = "transit"
            elif float(k.get("drop_height_est_m", 0)) > 0.3:
                state = "place"
            elif float(k.get("peak_jerk_mps3", 0)) > 5:
                state = "handle"
            else:
                state = "idle"
            steps.append((t, state, f"kinematic bucket {i}"))
        out.append(fsm.analyze(tid, steps).to_dict())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"{len(out)} FSM traces -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())