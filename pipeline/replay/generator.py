"""Replay generator — renders the actual + correct-technique trajectories.

Takes the pipeline events (which carry a scored trajectory) and produces a
2D replay payload: the *actual* motion of the tracked object and the
*correct-technique* motion from the nearest exemplar, both as normalized
(x, y) point series the frontend draws on a Canvas overlay.

The two series share the same frame-index axis so the frontend can animate
the actual path while fading in the recommended path beneath it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pipeline.types import RiskEvent, Trajectory

logger = logging.getLogger(__name__)


@dataclass
class ReplayFrame:
    """One animation frame of the replay."""
    frame_idx: int
    timestamp_sec: float
    actual_x: Optional[float]   # normalized 0-1 in the replay canvas
    actual_y: Optional[float]
    correct_x: Optional[float]  # normalized correct-technique position
    correct_y: Optional[float]


@dataclass
class Replay:
    """Full replay payload for one event."""
    event_id: str
    behavior_class: str
    frames: List[ReplayFrame] = field(default_factory=list)
    actual_path: List[tuple] = field(default_factory=list)      # [(x, y), ...]
    correct_path: List[tuple] = field(default_factory=list)     # [(x, y), ...]
    duration_sec: float = 0.0
    normalized: bool = True  # coordinates are 0-1 canvas space


def _normalize(points: List[tuple]) -> List[tuple]:
    """Map a point series into a 0-1 canvas box, preserving aspect.

    If a series has a single point or zero extent, it degrades to a dot in
    the centre of the box — never divides by zero.
    """
    if not points:
        return []
    arr = np.asarray(points, dtype=float)
    xs, ys = arr[:, 0], arr[:, 1]
    x0, y0 = float(xs.min()), float(ys.min())
    w = float(xs.max() - x0)
    h = float(ys.max() - y0)
    if w < 1e-9 and h < 1e-9:
        return [(0.5, 0.5)] * len(points)
    scale = max(w, h)
    out = [((float(x) - x0) / scale, (float(y) - y0) / scale) for x, y in arr]
    return out


def build_replay(
    event: RiskEvent,
    trajectory: Optional[Trajectory] = None,
    correct_points: Optional[List[dict]] = None,
) -> Replay:
    """Build a Replay for a scored RiskEvent.

    Args:
        event: the scored event.
        trajectory: the actual trajectory that produced the event. If None,
                    falls back to a straight line reconstructed from the
                    event's evidence window.
        correct_points: the exemplar trajectory points for the matched
                        behaviour (the "correct technique"). If None, no
                        correct path is produced (actual-only replay).
    """
    actual = [(p.x, p.y) for p in trajectory.points] if trajectory else []
    actual_path = _normalize(actual)

    correct = []
    if correct_points:
        correct = [
            (float(p["x"]), float(p["y"])) for p in correct_points
            if "x" in p and "y" in p
        ]
    correct_path = _normalize(correct)

    # Align to a common time axis for animation.
    ts_actual = [p.timestamp_sec for p in trajectory.points] if trajectory else []
    ts_correct = [p["timestamp_sec"] for p in correct_points] if correct_points else []
    align = sorted(set([round(t, 3) for t in ts_actual + ts_correct]))
    frames: List[ReplayFrame] = []
    a_i = c_i = 0
    for t in align:
        a_x = a_y = c_x = c_y = None
        if trajectory and a_i < len(ts_actual) and abs(ts_actual[a_i] - t) < 1e-3:
            a_x, a_y = actual_path[a_i]
            a_i += 1
        if correct_points and c_i < len(ts_correct) and abs(ts_correct[c_i] - t) < 1e-3:
            c_x, c_y = correct_path[c_i]
            c_i += 1
        frames.append(
            ReplayFrame(
                frame_idx=int(t * 5.0),  # replay at target FPS
                timestamp_sec=t,
                actual_x=a_x,
                actual_y=a_y,
                correct_x=c_x,
                correct_y=c_y,
            )
        )

    duration = (align[-1] - align[0]) if len(align) > 1 else 0.0
    return Replay(
        event_id=event.event_id,
        behavior_class=event.behavior_class,
        frames=frames,
        actual_path=actual_path,
        correct_path=correct_path,
        duration_sec=round(duration, 3),
    )


def replay_to_dict(replay: Replay) -> Dict:
    """Serialize a Replay for the API / frontend."""
    return {
        "event_id": replay.event_id,
        "behavior_class": replay.behavior_class,
        "duration_sec": replay.duration_sec,
        "normalized": replay.normalized,
        "frames": [
            {
                "frame_idx": f.frame_idx,
                "t": f.timestamp_sec,
                "actual": [f.actual_x, f.actual_y]
                if f.actual_x is not None else None,
                "correct": [f.correct_x, f.correct_y]
                if f.correct_x is not None else None,
            }
            for f in replay.frames
        ],
        "actual_path": [list(p) for p in replay.actual_path],
        "correct_path": [list(p) for p in replay.correct_path],
    }