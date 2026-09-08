"""Trajectory builder — converts frame-level tracked objects into
per-object time series.

Each trajectory is a ``(track_id, [(t, x, y, z_est), ...])`` series with
velocity, and is smoothed with either a moving average or a Kalman filter.

Output schema (JSON):
    {
        "track_id": int,
        "class_label": str,
        "points": [{"frame_idx", "timestamp_sec", "x", "y", "z_est", "vx", "vy"}],
        "total_displacement": float,
        "mean_speed": float,
        "max_speed": float
    }
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from pipeline.types import TrackedObject, Trajectory, TrajectoryPoint

logger = logging.getLogger(__name__)


class TrajectoryBuilder:
    """Accumulate tracked objects across frames into smoothed trajectories.

    Args:
        smoother: "moving_average" (default) or "kalman".
        ma_window: moving-average window size for position smoothing.
        min_track_length: trajectories shorter than this (frames) are dropped.
        pixel_scale: meters per pixel (from config physics calibration).
        max_gap: max frames a track can miss before the trajectory is sealed.
    """

    def __init__(
        self,
        smoother: str = "moving_average",
        ma_window: int = 5,
        min_track_length: int = 3,
        pixel_scale: float = 0.01,
        max_gap: int = 10,
        min_speed_mps: float = 0.05,
    ) -> None:
        if smoother not in {"moving_average", "kalman"}:
            raise ValueError(
                f"Unknown smoother {smoother!r}. Use 'moving_average' or 'kalman'."
            )
        self.smoother = smoother
        self.ma_window = ma_window
        self.min_track_length = min_track_length
        self.pixel_scale = pixel_scale
        self.max_gap = max_gap
        self.min_speed_mps = min_speed_mps

        self._raw: Dict[int, List[dict]] = {}   # track_id -> raw points
        self._tracks_meta: Dict[int, str] = {}  # track_id -> class_label
        self._sealed: List[Trajectory] = []     # finished trajectories
        self._age_map: Dict[int, int] = {}      # track_id -> consecutive misses

    # ------------------------------------------------------------------ public

    def reset(self) -> None:
        """Start a new video: clears accumulated state."""
        self._raw = {}
        self._tracks_meta = {}
        self._sealed = []

    def update(
        self,
        tracked: List[TrackedObject],
        frame_idx: int,
        timestamp_sec: float,
    ) -> None:
        """Add one frame of tracked objects.

        Call once per processed frame. Tracks missing from ``tracked`` are
        aged and sealed after ``max_gap`` consecutive misses.
        """
        seen: set = set()
        for obj in tracked:
            tid = obj.track_id
            seen.add(tid)
            cx, cy = obj.centroid or self._bbox_center(obj.bbox)
            if tid not in self._raw:
                self._raw[tid] = []
                self._tracks_meta[tid] = obj.class_label
            self._raw[tid].append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "x": cx * self.pixel_scale,
                    "y": cy * self.pixel_scale,
                    # z_est placeholder — refined by physics using pose/height
                    "z_est": 0.0,
                }
            )

        # Age unseen tracks; seal those past max_gap
        for tid, pts in list(self._raw.items()):
            if tid in seen:
                continue
            self._age_map.setdefault(tid, 0)
            self._age_map[tid] += 1
            if self._age_map[tid] > self.max_gap:
                self._seal(tid)
        for tid in seen:
            self._age_map.pop(tid, None)

    def finalize(self) -> List[Trajectory]:
        """Seal all remaining tracks and return the full trajectory list."""
        for tid in list(self._raw.keys()):
            self._seal(tid)
        self._raw = {}
        result = list(self._sealed)
        self._sealed = []
        # Sort by track_id for determinism
        result.sort(key=lambda t: t.track_id)
        return result

    # ------------------------------------------------------------------ private

    def _seal(self, tid: int) -> None:
        pts = self._raw.pop(tid, None)
        if not pts or len(pts) < self.min_track_length:
            return
        points = self._build_points(tid, pts)
        if len(points) < self.min_track_length:
            return
        traj = self._assemble(tid, points)
        # Skip near-static tracks — they're parked objects, not behaviours.
        if traj.max_speed < self.min_speed_mps and not traj.points[-1].z_est:
            return
        self._sealed.append(traj)

    def _build_points(self, tid: int, raw: List[dict]) -> List[TrajectoryPoint]:
        """Convert raw points to TrajectoryPoints with smoothing + velocity."""
        xs = np.array([p["x"] for p in raw], dtype=float)
        ys = np.array([p["y"] for p in raw], dtype=float)
        ts = np.array([p["timestamp_sec"] for p in raw], dtype=float)

        # Smoothing
        if self.smoother == "moving_average" and len(xs) >= self.ma_window:
            xs = self._moving_average(xs)
            ys = self._moving_average(ys)
        elif self.smoother == "kalman":
            xs, ys = self._kalman_filter(xs, ys)

        # Velocities (finite differences, clipped dt to avoid div-by-zero)
        dt = np.diff(ts)
        dt = np.where(dt <= 0, 1e-6, dt)
        vx = np.zeros_like(xs)
        vy = np.zeros_like(ys)
        vx[1:] = np.diff(xs) / dt
        vy[1:] = np.diff(ys) / dt

        points: List[TrajectoryPoint] = []
        for i, p in enumerate(raw):
            points.append(
                TrajectoryPoint(
                    frame_idx=int(p["frame_idx"]),
                    timestamp_sec=float(p["timestamp_sec"]),
                    x=float(xs[i]),
                    y=float(ys[i]),
                    z_est=self._estimate_z(raw, i),
                    vx=float(vx[i]),
                    vy=float(vy[i]),
                )
            )
        return points

    def _assemble(self, tid: int, points: List[TrajectoryPoint]) -> Trajectory:
        """Compute summary stats for a sealed trajectory."""
        pts = np.array([[p.x, p.y] for p in points])
        disp = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 1 else 0.0
        speeds = np.hypot([p.vx for p in points], [p.vy for p in points])
        return Trajectory(
            track_id=tid,
            points=points,
            class_label=self._tracks_meta.get(tid, "object"),
            total_displacement=disp,
            mean_speed=float(np.mean(speeds)) if len(speeds) else 0.0,
            max_speed=float(np.max(speeds)) if len(speeds) else 0.0,
        )

    @staticmethod
    def _bbox_center(bbox: List[int]) -> tuple:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _moving_average(arr: np.ndarray) -> np.ndarray:
        """Simple centered moving average preserving length."""
        window = min(len(arr), 5)
        if window < 2:
            return arr
        kernel = np.ones(window) / window
        padded = np.pad(arr, (window // 2, window // 2), mode="edge")
        out = np.convolve(padded, kernel, mode="valid")
        return out[: len(arr)]

    @staticmethod
    def _kalman_filter(xs: np.ndarray, ys: np.ndarray) -> tuple:
        """Lightweight 2D Kalman smoothing (numpy only)."""
        n = len(xs)
        state = np.array([xs[0], ys[0], 0.0, 0.0])  # [x, y, vx, vy]
        P = np.eye(4)
        F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        Q = np.eye(4) * 1e-4
        R = np.eye(2) * 1e-1
        out_x, out_y = [], []
        for i in range(n):
            state = F @ state
            P = F @ P @ F.T + Q
            z = np.array([xs[i], ys[i]])
            y = z - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            state = state + K @ y
            P = (np.eye(4) - K @ H) @ P
            out_x.append(state[0])
            out_y.append(state[1])
        return np.array(out_x), np.array(out_y)

    @staticmethod
    def _estimate_z(raw: List[dict], i: int) -> float:
        """Placeholder z-estimate: 0. Refined by physics stage using reference
        object geometry / pose. Kept for schema stability."""
        del raw, i
        return 0.0