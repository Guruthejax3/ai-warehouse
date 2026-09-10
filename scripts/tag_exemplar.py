#!/usr/bin/env python3
"""Tag a trajectory as an exemplar for behaviour matching.

Reads a trajectory JSON (the pipeline's trajectory output format) and
adds it to data/exemplars/behaviors.json under a behaviour class.

    usage:
      python scripts/tag_exemplar.py trajectories.json \
          --class product_dropped \
          --label "Real drop of glass jars, bay 3"

    --class: one of:
      product_dropped, product_dragged, rough_handling, incorrect_stacking,
      unstable_stacking, outside_designated_zone, no_required_equipment,
      pallet_mispositioned, material_pushed_or_thrown, unsafe_loading_sequence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.matching.dtw_matcher import ExemplarLibrary
from pipeline.types import BEHAVIOR_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory_json",
        type=Path,
        help="Trajectory JSON file (pipeline output: list of trajectory dicts)",
    )
    parser.add_argument(
        "--track-id", type=int, default=None,
        help="Which track in the file to tag (default: first)",
    )
    parser.add_argument(
        "--class", dest="behavior_class", required=True,
        choices=BEHAVIOR_CLASSES,
        help="Behaviour class to assign",
    )
    parser.add_argument(
        "--label", default="", help="Human-readable label for the exemplar",
    )
    parser.add_argument(
        "--library", default="data/exemplars/behaviors.json",
        help="Exemplar library path",
    )
    parser.add_argument(
        "--source-clip", default=None, help="Source clip filename",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.trajectory_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    trajectories = data if isinstance(data, list) else data.get("trajectories", [])
    if not trajectories:
        print("ERROR: no trajectories found in file.", file=sys.stderr)
        return 1

    selected = trajectories[0]
    if args.track_id is not None:
        selected = next(
            (t for t in trajectories if t.get("track_id") == args.track_id), None
        )
        if selected is None:
            print(f"ERROR: track {args.track_id} not found.", file=sys.stderr)
            return 1

    points = selected.get("points", [])
    if not points:
        print("ERROR: selected trajectory has no points.", file=sys.stderr)
        return 1

    library = ExemplarLibrary(args.library)
    exemplar_id = library.add_exemplar(
        behavior_class=args.behavior_class,
        label=args.label or selected.get("class_label", args.behavior_class),
        points=points,
        source_clip=args.source_clip,
        metadata={
            "track_id": selected.get("track_id"),
            "total_displacement": selected.get("total_displacement"),
            "mean_speed": selected.get("mean_speed"),
            "max_speed": selected.get("max_speed"),
        },
    )
    print(f"Added exemplar: {exemplar_id}")
    print(f"  behavior_class = {args.behavior_class}")
    print(f"  label          = {args.label}")
    print(f"  points         = {len(points)}")
    print(f"  library        = {library.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())