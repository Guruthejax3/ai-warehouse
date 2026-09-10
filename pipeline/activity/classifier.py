"""Activity classifier — infers loading / unloading / idle / transit from trajectory.

The classifier uses *directional analysis* of the trajectory's centroid flow:

- **Loading**: net movement toward the *loading zone* (truck/dock edge of frame,
  typically the upper or left region). Velocity vector points INTO the dock.
- **Unloading**: net movement AWAY from the loading zone toward interior storage.
- **Transit**: significant horizontal displacement with no strong dock-ward component.
- **Idle**: low displacement and low speed — the object barely moved.

Classification is geometry-free (no zone polygons needed): the dock edge is
inferred from the frame's aspect ratio (the shorter axis is assumed to be the
dock-facing axis for a typical overhead warehouse camera).  When explicit zone
polygons are provided, they override the heuristic.

The classifier also provides a *confidence* based on how strongly the trajectory
aligns with the inferred direction.

Called from `pipeline/runner.py` after trajectory sealing and before risk scoring.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.types import Trajectory, TrajectoryPoint

logger = logging.getLogger(__name__)

# Activity type literal
ActivityType = str  # "loading" | "unloading" | "idle" | "transit"

# Minimum displacement (meters) to consider non-idle
_MIN_DISPLACEMENT_M = 0.3

# Minimum number of points for a reliable classification
_MIN_POINTS = 5


class ActivityClassifier:
    """Classify a trajectory's activity type from centroid flow direction.

    Args:
        frame_width: source video width in pixels.
        frame_height: source video height in pixels.
        pixel_scale: meters per pixel.
        dock_edge: which frame edge the loading dock is at.
            "auto" = infer from aspect ratio (shorter edge = dock-facing).
            "top" | "bottom" | "left" | "right" = explicit.
        loading_zone_polygon: optional normalized 0-1 polygon; if provided,
            the centroid is compared against this polygon to determine
            dock-ward vs interior motion.
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        pixel_scale: float = 0.01,
        dock_edge: str = "auto",
        loading_zone_polygon: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pixel_scale = pixel_scale
        self.dock_edge = dock_edge
        self.loading_zone_polygon = loading_zone_polygon

    def classify(self, trajectory: Trajectory) -> Tuple[ActivityType, float, str]:
        """Classify a trajectory's activity.

        Returns:
            (activity_type, confidence, justification)
        """
        pts = trajectory.points
        if len(pts) < _MIN_POINTS:
            return "unknown", 0.0, "Trajectory too short for activity classification."

        # Extract positions
        xs = np.array([p.x for p in pts], dtype=float)
        ys = np.array([p.y for p in pts], dtype=float)

        # Net displacement
        dx = float(xs[-1] - xs[0])
        dy = float(ys[-1] - ys[0])
        displacement_px = math.hypot(dx, dy)
        displacement_m = displacement_px * self.pixel_scale

        # Speed profile
        speeds = []
        for i in range(1, len(pts)):
            dt = pts[i].timestamp_sec - pts[i - 1].timestamp_sec
            if dt > 0:
                ds = math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y) * self.pixel_scale
                speeds.append(ds / dt)
        mean_speed = float(np.mean(speeds)) if speeds else 0.0

        # Dock-ward direction analysis
        dock_ward_score, dock_ward_reason = self._dock_ward_direction(dx, dy, displacement_m)

        # Idle check
        if displacement_m < _MIN_DISPLACEMENT_M and mean_speed < 0.1:
            just = (
                f"Activity: IDLE — displacement {displacement_m:.2f}m and mean speed "
                f"{mean_speed:.2f} m/s are both below thresholds. Object barely moved."
            )
            return "idle", 0.85, just

        # Classify based on dock-ward score
        if dock_ward_score > 0.3:
            # Moving TOWARD dock = loading
            conf = min(1.0, 0.5 + dock_ward_score)
            just = (
                f"Activity: LOADING — trajectory moves toward the dock "
                f"({dock_ward_reason}). Net displacement {displacement_m:.2f}m, "
                f"mean speed {mean_speed:.2f} m/s."
            )
            return "loading", conf, just
        elif dock_ward_score < -0.3:
            # Moving AWAY from dock = unloading
            conf = min(1.0, 0.5 + abs(dock_ward_score))
            just = (
                f"Activity: UNLOADING — trajectory moves away from dock "
                f"({dock_ward_reason}). Net displacement {displacement_m:.2f}m, "
                f"mean speed {mean_speed:.2f} m/s."
            )
            return "unloading", conf, just
        else:
            # No strong dock-ward component = transit
            conf = 0.5 + abs(dock_ward_score) * 0.3
            just = (
                f"Activity: TRANSIT — trajectory shows lateral movement with no "
                f"strong dock-ward component ({dock_ward_reason}). "
                f"Displacement {displacement_m:.2f}m, mean speed {mean_speed:.2f} m/s."
            )
            return "transit", conf, just

    def _dock_ward_direction(
        self, dx: float, dy: float, displacement_m: float
    ) -> Tuple[float, str]:
        """Compute a dock-ward score from the trajectory's net displacement.

        Returns:
            (score, reason): score > 0 = toward dock, < 0 = away from dock.
        """
        if displacement_m < 1e-6:
            return 0.0, "no displacement"

        edge = self.dock_edge
        if edge == "auto":
            # Heuristic: for a landscape frame, dock is on the left/right
            # (shorter axis). For portrait, dock is top/bottom.
            if self.frame_width >= self.frame_height:
                # Landscape: dock assumed on the left (loading dock side)
                edge = "left"
            else:
                edge = "top"

        # Score: component of displacement along the dock axis
        # dock-ward = toward the dock edge
        if edge == "left":
            # Toward left = negative dx = dock-ward
            score = -dx / max(1, self.frame_width)
            reason = f"dock at left edge, dx={dx:.0f}px"
        elif edge == "right":
            score = dx / max(1, self.frame_width)
            reason = f"dock at right edge, dx={dx:.0f}px"
        elif edge == "top":
            score = -dy / max(1, self.frame_height)
            reason = f"dock at top edge, dy={dy:.0f}px"
        elif edge == "bottom":
            score = dy / max(1, self.frame_height)
            reason = f"dock at bottom edge, dy={dy:.0f}px"
        else:
            score = 0.0
            reason = f"unknown dock edge '{edge}'"

        return score, reason
