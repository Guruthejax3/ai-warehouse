"""Deterministic layout rules for behaviours the spec assigns to RULES, not DTW.

Several of the 10 behaviours are rule-driven by design (see README / spec), so
they must NOT compete in the DTW exemplar library — otherwise a generic
straight-line exemplar swallows every translatory track (a floor drag was being
reported as an outside-zone violation because both are "long straight-ish").

Implemented here:
    outside_designated_zone  — trajectory's final centroid leaves a configured
                               zone polygon (per camera, normalized 0-1 coords).

Stubbed (TODO — need detectors/state machines, see README "known limitations"):
    incorrect_stacking       — needs mask-size ordering of a stack cluster
    unstable_stacking        — needs stack-cluster centroid drift/tilt tracking
    no_required_equipment    — needs equipment classification + proximity radius
    pallet_mispositioned     — needs pallet/dock-mark geometry comparison
    unsafe_loading_sequence  — needs an expected-order FSM

Zone polygons live in config.yaml:
    zones:
      default:                       # fallback when camera id is unknown
        - [[0.0, 0.0], [1.0, 0.0], [1.0, 0.9], [0.0, 0.9]]
      bay_1: [[0.2, 0.1], [0.9, 0.1], [0.9, 0.8], [0.2, 0.8]]
Coordinates are normalized 0-1 (independent of resolution / pixel-scale), so
the same config works across cameras. A track whose final centroid is outside
every polygon triggers an `outside_designated_zone` event.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from pipeline.types import Trajectory

logger = logging.getLogger(__name__)

#: Normalized polygon that covers everything (rule dormant until real polygons
#: are configured per camera — no false "outside zone" on bare footage).
_FULL_FRAME = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


class ZoneRuleChecker:
    """Evaluates the rule-based behaviours that depend on spatial layout.

    Args:
        zones_cfg: dict from config.yaml `zones:` — camera name -> list of
                   normalized polygons (each a list of [x, y] 0-1 vertices).
                   A `default` entry is used when the camera id is unknown.
    """

    def __init__(self, zones_cfg: Optional[Dict[str, Any]] = None) -> None:
        zones_cfg = zones_cfg or {}
        self.polygons: Dict[str, List[np.ndarray]] = {}
        for name, polys in zones_cfg.items():
            clean = []
            for poly in polys:
                arr = np.asarray(poly, dtype=float)
                if arr.size == 0:
                    continue
                clean.append(arr.reshape(-1, 2))
            if clean:
                self.polygons[name] = clean

    def check(
        self,
        trajectory: Trajectory,
        frame_w: int,
        frame_h: int,
        pixel_scale: float,
        camera_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Return an outside-zone event dict if the trajectory ends outside all
        configured polygons, else None.

        ``{behavior_class, reason, severity}`` — severity is a 0-1 confidence
        the risk scorer folds into the zone-criticality term.
        """
        polys = self.polygons.get(camera_id) or self.polygons.get("default")
        if not polys:
            # No zone config for this camera (and no fallback) — rule dormant.
            return None

        centroid = np.asarray(
            [trajectory.points[-1].x, trajectory.points[-1].y], dtype=float
        )
        # Trajectory stores meters (cx * pixel_scale) → recover pixels → normalize.
        px = centroid / max(pixel_scale, 1e-9)
        norm = np.array([px[0] / max(frame_w, 1), px[1] / max(frame_h, 1)])

        inside_any = any(
            cv2.pointPolygonTest(
                poly.astype(np.float32), tuple(map(float, norm)), False
            )
            >= 0
            for poly in polys
        )
        if inside_any:
            return None

        # How far outside, as a 0-1 severity (distance normalized by ~15% of frame).
        dist = min(
            float(
                cv2.pointPolygonTest(
                    poly.astype(np.float32), tuple(map(float, norm)), True
                )
            )
            for poly in polys
        )
        severity = float(np.clip(-dist / 0.15, 0.0, 1.0))
        n_pts = len(trajectory.points)
        return {
            "behavior_class": "outside_designated_zone",
            "reason": (
                f"Final centroid ({centroid[0]:.2f}, {centroid[1]:.2f}) m lies "
                f"outside the designated {camera_id} zone polygon (severity "
                f"{severity:.2f}, {n_pts} tracked points)."
            ),
            "severity": severity,
        }