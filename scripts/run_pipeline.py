#!/usr/bin/env python3
"""Run one warehouse clip end-to-end through the full ReplayTwin pipeline.

Chain: motion detection -> segmentation/tracking -> trajectory -> behaviour
matching (DTW) -> physics cross-check -> risk scoring.

    usage: python scripts/run_pipeline.py "data/clips/Throwing Mattresses.mp4"

Thin CLI wrapper over pipeline.runner.process_clip so the backend reuses the
exact same chain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.runner import process_clip  # noqa: E402

RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a warehouse clip")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N processed frames (0 = all)")
    parser.add_argument("--min-level", type=str, default="low",
                        choices=["low", "medium", "high", "critical"],
                        help="Only print events at this level or above")
    args = parser.parse_args()

    floor = RANK[args.min_level]
    events, _traj = process_clip(args.video, max_frames=args.max_frames)

    shown = [e for e in events if RANK[e.risk_level] >= floor]
    print(f"Processed {args.video.name}: {len(events)} scored events "
          f"(showing >= {args.min_level}: {len(shown)})")
    for ev in shown:
        flag = "gated" if ev.metadata.get("gated_by_physics") else "ALERT"
        print(
            f"  [{ev.risk_level.upper():8s}] {ev.behavior_class:28s} "
            f"score={ev.risk_score:.2f} {flag:6s} track={ev.trajectory_id}"
        )
        if ev.risk_level in ("high", "critical"):
            print(f"      -> {ev.justification[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())