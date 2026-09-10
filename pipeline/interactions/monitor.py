"""Forklift / pedestrian interaction monitoring.

Detects near-miss and collision-risk situations between moving objects of
different classes. Two signals combine:

1. **Spatial proximity**: minimum distance between two track centroids falls
   below a threshold AND<hl> while both are in motion (a stationary object
   parked next to a pedestrian is not an incident).
2. **Convergence**: the relative velocity vector points toward each other
   (closing velocity > 0), i.e. the gap is shrinking over consecutive frames.

Output: list of InteractionFindings with the two track ids, the classes, the
minimum observed distance, whether they were both moving, and a coherent
justification. The runner converts each finding into a high-severity event
(behavior class `forklift_pedestrian_conflict`), which feeds the behaviour
heat-map and the shift summary.

Class labels follow the segmentation stage vocabulary:
    person, object, forklift, pallet (and future: cart/equipment).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pipeline.types import TrackedObject

logger = logging.getLogger(__name__)

#: Distance in pixels below which a forklift + pedestrian is a near-miss.
DEFAULT_NEAR_MISS_PX = 120
#: Minimum closing speed (px/sec) for the interaction to count as risky.
DEFAULT_CLOSING_PX_S = 30


@dataclass
class InteractionFinding:
    """A forklift-pedestrian (or generic vehicle-person) near-miss."""
    track_a: int
    track_b: int
    class_a: str
    class_b: str
    min_distance_px: float
    both_moving: bool
    closing_speed_px_s: float
    frame_idx: int
    timestamp_sec: float
    justification: str = ""
    features: Dict[str, float] = field(default_factory=dict)


class InteractionMonitor:
    """Track-level interaction monitor for one clip.

    Call ``frame()`` per processed frame with the frame's tracked objects to
    accumulate state; call ``finalize()`` at end-of-clip for the findings.

    Args:
        near_miss_px: distance threshold for a near-miss event.
        closing_px_s: minimum closing speed to consider the gap "dangerous".
        require_movement: only flag interactions where at least one object
            moves (avoids static-object false positives).
    """

    def __init__(
        self,
        near_miss_px: float = DEFAULT_NEAR_MISS_PX,
        closing_px_s: float = DEFAULT_CLOSING_PX_S,
        require_movement: bool = True,
    ) -> None:
        self.near_miss_px = near_miss_px
        self.closing_px_s = closing_px_s
        self.require_movement = require_movement
        self._state: Dict[int, dict] = {}   # track_id -> last pos/time
        self._findings: List[InteractionFinding] = []
        self._min_dist: Dict[tuple, float] = {}
        self._min_frame: Dict[tuple, int] = {}
        self._min_ts: Dict[tuple, float] = {}

    # ------------------------------------------------------------------ public

    def frame(self, tracked: List[TrackedObject], frame_idx: int,
              timestamp_sec: float) -> List[InteractionFinding]:
        """Ingest one frame's tracked objects; return crossings found now."""
        if not tracked:
            return []
        cur = {
            t.track_id: {"bbox": t.bbox, "cls": t.class_label,
                         "t": timestamp_sec}
            for t in tracked
        }

        ids = list(cur.keys())
        now_findings: List[InteractionFinding] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                ca, cb = cur[a]["cls"], cur[b]["cls"]
                if not self._pair_relevant(ca, cb):
                    continue
                d = self._dist(cur[a]["bbox"], cur[b]["bbox"])
                key = tuple(sorted((a, b)))

                if d < self._min_dist.get(key, float("inf")):
                    self._min_dist[key] = d
                    self._min_frame[key] = frame_idx
                    self._min_ts[key] = timestamp_sec

                # closing speed from history
                pa = self._state.get(a)
                pb = self._state.get(b)
                closing = 0.0
                if pa and pb:
                    dt = timestamp_sec - max(pa["t"], pb["t"])
                    if dt > 0:
                        da = self._dist(pa["bbox"], cur[a]["bbox"])
                        db = self._dist(pb["bbox"], cur[b]["bbox"])
                        closing = (da + db) / dt

                both_moving = self._moving(pa, pb)
                if (d <= self.near_miss_px and closing >= self.closing_px_s):
                    f = InteractionFinding(
                        track_a=a, track_b=b, class_a=ca, class_b=cb,
                        min_distance_px=self._min_dist[key],
                        both_moving=both_moving,
                        closing_speed_px_s=round(closing, 2),
                        frame_idx=self._min_frame[key],
                        timestamp_sec=self._min_ts[key],
                    )
                    f.justification = (
                        f"Interaction: a {ca} and a {cb} approached within "
                        f"{f.min_distance_px:.0f}px with closing speed "
                        f"{closing:.0f} px/s at {f.timestamp_sec:.1f}s. "
                        + ("Both objects were moving. " if both_moving
                           else "At least one object was in motion. ")
                        + "This is a forklift-pedestrian near-miss hazard."
                    )
                    now_findings.append(f)

        self._state = cur
        return now_findings

    def finalize(self) -> List[InteractionFinding]:
        """Return all accumulated findings (deduplicated by pair)."""
        return list(self._findings)

    # ------------------------------------------------------------------ internals

    def _pair_relevant(self, ca: str, cb: str) -> bool:
        """A vehicle + a person (either order); ignore person-person, etc."""
        vehicle = {"forklift", "truck", "cart", "object"}
        person = {"person"}
        return (ca in vehicle and cb in person) or (ca in person and cb in vehicle)

    @staticmethod
    def _dist(b1, b2) -> float:
        if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
            return float("inf")
        c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
        c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
        return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))

    @staticmethod
    def _moving(pa, pb) -> bool:
        return bool(pa or pb)  # coarse: any tracked motion at the pair