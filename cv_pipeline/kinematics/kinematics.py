"""Kinematics Module — motion parameters from tracked objects.

Stage 4 of the Damage-DNA cv-pipeline. Consumes the tracking JSON from
stage 2 (per-track center points over time) and computes physical quantities:
velocity, acceleration, jerk, drop-height estimate, collision/environment
trajectory risk. All pixel-space values are converted to SI units via
``pixel_scale`` (metres per pixel).

Handoff contract (output JSON):
    {
        "track_id": int,
        "sample_count": int,
        "peak_speed_mps": float,
        "mean_speed_mps": float,
        "peak_accel_mps2": float,
        "peak_jerk_mps3": float,
        "drop_height_est_m": float,
        "vertical_thrust_mps": float,
        "stop_latency_ms": float,
        "collision_warning": bool,
        "severity_score": float,
        "justification": str
    }

severity_score (0-1) is the input to the risk_scoring stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class KinematicSummary:
    """Computed kinematic quantities for one tracked object."""
    track_id: int
    class_label: str = "object"
    sample_count: int = 0
    peak_speed_mps: float = 0.0
    mean_speed_mps: float = 0.0
    peak_accel_mps2: float = 0.0
    peak_jerk_mps3: float = 0.0
    drop_height_est_m: float = 0.0
    vertical_thrust_mps: float = 0.0      # max upward (away from floor) speed
    stop_latency_ms: float = 0.0          # time from peak speed to near-rest
    collision_warning: bool = False       # another object on a crossing course
    severity_score: float = 0.0           # 0..1 snapshot — feeds risk scoring
    justification: str = ""

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_label": self.class_label,
            "sample_count": self.sample_count,
            "peak_speed_mps": round(self.peak_speed_mps, 4),
            "mean_speed_mps": round(self.mean_speed_mps, 4),
            "peak_accel_mps2": round(self.peak_accel_mps2, 4),
            "peak_jerk_mps3": round(self.peak_jerk_mps3, 4),
            "drop_height_est_m": round(self.drop_height_est_m, 4),
            "vertical_thrust_mps": round(self.vertical_thrust_mps, 4),
            "stop_latency_ms": round(self.stop_latency_ms, 2),
            "collision_warning": self.collision_warning,
            "severity_score": round(self.severity_score, 4),
            "justification": self.justification,
        }


class KinematicsAnalyzer:
    """Compute motion physics from per-track center time-series.

    Args:
        pixel_scale: metres per pixel (image-to-world conversion).
        gravity: gravitational acceleration (m/s^2), used to normalise jerk.
        heading_service: optional callable mapping (x, y, ts) to a heading
            label; if omitted, heading is ignored. Kept for plugin symmetry
            with the activity classifier in ``pipeline/``.
    """

    def __init__(
        self,
        pixel_scale: float = 0.01,
        gravity: float = 9.81,
        heading_service: Optional[object] = None,
    ) -> None:
        self.pixel_scale = pixel_scale
        self.gravity = gravity
        self.heading_service = heading_service

    # ------------------------------------------------------------------ public

    def analyze(
        self,
        track_id: int,
        xs: List[float],
        ys: List[float],
        ts: List[float],
        class_label: str = "object",
        other_tracks: Optional[Dict[int, Tuple[List[float], List[float], List[float]]]] = None,
        frame_w: int = 1280,
        frame_h: int = 720,
    ) -> KinematicSummary:
        """Compute a kinematic summary for one track's trajectory.

        Args:
            track_id: the track id.
            xs, ys: pixel x/y positions per sample.
            ts: timestamps (seconds) per sample.
            class_label: "person" | "object" | ...
            other_tracks: map track_id -> (xs, ys, ts) for collision checks.
            frame_w, frame_h: frame size (for drop-height ceiling bound).
        """
        if len(xs) < 2:
            return KinematicSummary(
                track_id=track_id, class_label=class_label,
                justification="Insufficient samples (<2) to compute kinematics.",
            )

        ps = self.pixel_scale or 0.01
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        ts = np.asarray(ts, dtype=float)
        dt = np.diff(ts)
        dt = np.maximum(dt, 1e-4)

        vx = np.diff(xs) / dt * ps
        vy = np.diff(ys) / dt * ps
        speed = np.hypot(vx, vy)

        ax = np.diff(vx) / np.maximum(dt[1:], 1e-4)
        ay = np.diff(vy) / np.maximum(dt[1:], 1e-4)
        accel = np.hypot(ax, ay)

        jx = np.diff(ax) / np.maximum(dt[2:], 1e-4)
        jy = np.diff(ay) / np.maximum(dt[2:], 1e-4)
        jerk = np.hypot(jx, jy)

        peak_speed = float(speed.max()) if len(speed) else 0.0
        mean_speed = float(speed.mean()) if len(speed) else 0.0
        peak_accel = float(accel.max()) if len(accel) else 0.0
        peak_jerk = float(jerk.max()) if len(jerk) else 0.0

        # Drop height: (peak - min) of vertical position, minus a camera-roll
        # clamp so tall jumps don't inflate it; bounded by frame size.
        span_px = float(ys.max() - ys.min())
        drop_m = span_px * ps
        drop_m = min(drop_m, frame_h * ps * 0.9)

        # Vertical thrust: fastest upward motion (negative screen-y = upward
        # toward the ceiling in a top-down-ish dock view is 'away'; here upward
        # in world = toward top of frame = -vy, so thrust = max(-vy)). We use
        # the magnitude of the most vertical velocity spike.
        vertical = float(np.min(vy)) if len(vy) else 0.0
        thrust_mps = max(0.0, -vertical)

        # Stop latency: time between peak speed and dropping to rest.
        stop_latency = 0.0
        if peak_speed > 0.05:
            peak_idx = int(speed.argmax())
            rest_thresh = max(0.1, 0.2 * peak_speed)
            rest_idx = peak_idx
            while rest_idx < len(speed) and speed[rest_idx] > rest_thresh:
                rest_idx += 1
            stop_latency = float((ts[min(rest_idx, len(ts) - 1)] - ts[peak_idx]) * 1000.0)

        # Collision warning: another track whose path crosses this one's bbox
        # window within the frame (simple min-distance over the overlap).
        collision = False
        if other_tracks:
            for other_id, (oxs, oys, _ots) in other_tracks.items():
                if other_id == track_id:
                    continue
                oxs = np.asarray(oxs, dtype=float)
                oys = np.asarray(oys, dtype=float)
                n = min(len(xs), len(oxs))
                if n < 2:
                    continue
                dist = np.hypot(xs[-n:] - oxs[-n:], ys[-n:] - oys[-n:]) * ps
                if float(dist.min()) < 0.35:
                    collision = True

        # Severity: weighted physics snapshot in [0,1].
        s_speed = min(1.0, peak_speed / 1.5)
        s_jerk = min(1.0, peak_jerk / (self.gravity * 2.0))
        s_height = min(1.0, drop_m / 1.0)
        severity = 0.35 * s_speed + 0.35 * s_jerk + 0.20 * s_height
        if collision:
            severity = min(1.0, severity + 0.15)

        reasons = []
        if peak_speed > 1.0:
            reasons.append(f"peak speed {peak_speed:.2f} m/s")
        if peak_jerk > self.gravity * 0.8:
            reasons.append(f"peak jerk {peak_jerk:.1f} m/s^3")
        if drop_m > 0.4:
            reasons.append(f"estimated drop height {drop_m:.2f} m")
        if stop_latency > 250:
            reasons.append(f"slow stop latency {stop_latency:.0f} ms")
        if collision:
            reasons.append("crossing path with another object")
        justification = (
            "Kinematics: " + (", ".join(reasons) if reasons else "within normal range") + "."
        )

        return KinematicSummary(
            track_id=track_id,
            class_label=class_label,
            sample_count=len(xs),
            peak_speed_mps=round(peak_speed, 4),
            mean_speed_mps=round(mean_speed, 4),
            peak_accel_mps2=round(peak_accel, 4),
            peak_jerk_mps3=round(peak_jerk, 4),
            drop_height_est_m=round(drop_m, 4),
            vertical_thrust_mps=round(thrust_mps, 4),
            stop_latency_ms=round(stop_latency, 2),
            collision_warning=collision,
            severity_score=round(severity, 4),
            justification=justification,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def trajectories_from_tracking(tracking_frames: List[dict]) -> Dict[int, dict]:
        """Extract per-track (xs, ys, ts, class_label) from tracking JSON.

        This is the seam between stage 2 (detection_tracking) and this stage.
        """
        trajs: Dict[int, dict] = {}
        for frame in tracking_frames:
            fidx = int(frame["frame_idx"])
            ts = float(frame.get("timestamp_sec", 0.0))
            for t in frame.get("tracks", []):
                if t.get("bbox") is None:
                    continue
                tid = int(t["track_id"])
                cx, cy = t["center"]
                d = trajs.setdefault(tid, {
                    "xs": [], "ys": [], "ts": [], "class_label": t.get("class_label", "object")
                })
                d["xs"].append(cx)
                d["ys"].append(cy)
                d["ts"].append(ts)
        return trajs