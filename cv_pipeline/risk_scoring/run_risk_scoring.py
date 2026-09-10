"""CLI for the Risk Scoring stage. Merges kinematics + FSM JSON, writes events.

Usage:
    python -m cv_pipeline.risk_scoring.run_risk_scoring \
        --kinematics kinematics.json --fsm fsm.json --output events.json
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from cv_pipeline.risk_scoring.risk_scoring import RiskScorer


def main() -> int:
    p = argparse.ArgumentParser(description="Risk Scoring stage.")
    p.add_argument("--kinematics", type=Path, required=True)
    p.add_argument("--fsm", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("events.json"))
    args = p.parse_args()

    kdata = json.loads(args.kinematics.read_text())
    fdata = json.loads(args.fsm.read_text())
    scorer = RiskScorer()
    out = []
    for k in kdata:
        tid = int(k["track_id"])
        fsm = next((f for f in fdata if int(f["track_id"]) == tid), {})
        ev = scorer.score(
            event_id=uuid.uuid4().hex[:12],
            track_id=tid,
            timestamp_sec=float(k.get("sample_count", 0) or 0) / 10.0,
            kinematics_severity=float(k.get("severity_score", 0)),
            pose_hazard=False,
            fsm_unsafe=bool(fsm.get("violations")),
            drop_height_m=float(k.get("drop_height_est_m", 0)),
            peak_speed_mps=float(k.get("peak_speed_mps", 0)),
        )
        out.append(ev.to_dict())
    out.sort(key=lambda e: -e["risk_score"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"{len(out)} scored events -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())