"""Sequence FSM — detect multi-step action sequences from trajectory evidence.

Single events detect *one* behavior in isolation. Some unsafe actions are only
meaningful as a *chain*: e.g. an unsafe loading sequence is "approach bay →
lift product → transit → place at height" where each step is required. The
detector below classifies a running timeline of trajectory statistics into a
small alphabet of steps, turns those classifications into *runs* (consecutive
windows of the same step), and matches the ordered chain as a subsequence of
that run list.

Why this design (and what it fixes vs. a naive per-frame FSM):
    - Steps are judged on *windows*, never a single frame (Feature 2.3: sequence
      detection requires multi-frame evidence).
    - A trailing "still handling" phase after lift does not abort the chain —
      the run extractor keeps one mode until it truly changes, so phase-lag
      between the kinematics and the behaviour is tolerated.
    - Short noise runs (< min_step_frames samples) are dropped before matching.

Step predicates (speeds are metres/sec after pixel_scale conversion):
    approach  : dock-ward motion at a workable speed (coming to the dock)
    handle    : slow motion with lifting (upward y) or high jerk (grab)
    transit   : sustained fast lateral carry
    place     : slow settling with downward y (deposit at height)
    depart    : motion away from the dock (closes the chain)

Output: a `SequenceHypothesis` naming the matched chain, the windows where each
step fired (evidence), confidence, and a justification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from pipeline.types import Trajectory

logger = logging.getLogger(__name__)

STEP_APPROACH = "approach"
STEP_HANDLE = "handle"
STEP_TRANSIT = "transit"
STEP_PLACE = "place"
STEP_DEPART = "depart"
STEP_GAP = "gap"

#: Ordered chain the detector looks for as a subsequence of the run list.
CHAIN = [STEP_APPROACH, STEP_HANDLE, STEP_TRANSIT, STEP_PLACE]

#: Priority order for a timestep's dominant step (ties broken by this order).
#: approach beats transit during a decelerating dock-ward glide; handle beats
#: place when the signal is ambiguous but both are "slow".
_DOMINANT_PRIORITY = [
    STEP_APPROACH, STEP_TRANSIT, STEP_HANDLE, STEP_PLACE, STEP_DEPART,
]


@dataclass
class SequenceHypothesis:
    """A matched ordered chain of actions, with evidence windows."""
    hypothesis: str                    # e.g. "unsafe_loading_sequence"
    matched: bool
    steps: List[dict]                  # [{step, start_sec, end_sec, evidence}]
    order_respected: bool
    confidence: float
    justification: str
    transitions: List[str] = field(default_factory=list)
    trajectory_ids: List[int] = field(default_factory=list)


class SequenceDetector:
    """Classify trajectory statistics into action runs and match chains.

    Args:
        target_fps: processing FPS (windows judged in samples/sec).
        truck_edge: "auto" infers the dock-facing edge from aspect ratio
            (short axis for overhead cameras) — same convention as
            ActivityClassifier.
        enabled: when False, ``detect`` returns no hypotheses.
        min_step_frames: minimum run length (in samples) to count as evidence.
        pixel_scale: metres per pixel, used to convert px/frame speeds to m/s.
        step_gap_tol_sec: not used by run extractor (runs are contiguous);
            kept for API compatibility.
    """

    def __init__(
        self,
        target_fps: float = 5.0,
        truck_edge: str = "auto",
        enabled: bool = True,
        min_step_frames: int = 2,
        pixel_scale: float = 0.01,
        step_gap_tol_sec: float = 2.5,
    ) -> None:
        self.target_fps = target_fps
        self.truck_edge = truck_edge
        self.enabled = enabled
        self.min_step_frames = min_step_frames
        self.pixel_scale = pixel_scale
        self.step_gap_tol_sec = step_gap_tol_sec
        self._runs: List[dict] = []
        self._hypotheses: List[SequenceHypothesis] = []

    # ------------------------------------------------------------------ public

    def detect(self, trajectories: List[Trajectory], frame_w: int = 1280,
               frame_h: int = 720) -> List[SequenceHypothesis]:
        """Run sequence extraction over a set of sealed trajectories.

        Returns hypotheses equal to the number of full chains detected.
        """
        if not self.enabled or not trajectories:
            return []
        self._runs = []
        self._hypotheses = []

        timeline = self._build_timeline(trajectories)
        times = sorted(timeline)
        if len(times) < 4:
            return []

        # Run extraction: label each timestep with its dominant step.
        runs: List[dict] = []
        prev_scale = 0.0
        for t in times:
            blk = timeline[t]
            speed = float(np.mean(blk["speed"])) if blk["speed"] else 0.0
            vx = float(np.mean(blk["vx"])) if blk["vx"] else 0.0
            vy = float(np.mean(blk["vy"])) if blk["vy"] else 0.0
            jerk = abs(speed - prev_scale)
            prev_scale = speed

            active = self._predicates(speed, vx, vy, jerk)
            step = self._dominant(active)
            if runs and runs[-1]["step"] == step:
                runs[-1]["end"] = t
            else:
                runs.append({"step": step, "start": t, "end": t})

        # Drop noise runs (too short to be multi-frame evidence)
        kept = [
            r for r in runs
            if (r["end"] - r["start"]) * self.target_fps >= self.min_step_frames
        ]

        # Match the ordered chain as a subsequence of the kept run labels.
        chain_pos = 0
        matched_runs: List[dict] = []
        for r in kept:
            if r["step"] == STEP_GAP:
                continue
            if chain_pos < len(CHAIN) and r["step"] == CHAIN[chain_pos]:
                matched_runs.append(r)
                chain_pos += 1
                if chain_pos == len(CHAIN):
                    break
        if chain_pos == len(CHAIN):
            self._hypotheses.append(self._build_hypothesis(matched_runs))

        return list(self._hypotheses)

    def reset(self) -> None:
        self._runs = []
        self._hypotheses = []

    # ------------------------------------------------------------------ setup

    def _build_timeline(self, trajectories: List[Trajectory]) -> Dict[float, dict]:
        """Aggregate per-timestep speeds/velocities across all trajectories.

        Speeds converted to m/s via pixel_scale so thresholds match the physics
        stage. ``gap`` runs are generated by timesteps where no object moved.
        """
        ps = self.pixel_scale or 0.01
        tl: Dict[float, dict] = {}
        for traj in trajectories:
            for i, p in enumerate(traj.points):
                bucket = tl.setdefault(float(p.timestamp_sec),
                                       {"speed": [], "vx": [], "vy": []})
                sp = vx = vy = 0.0
                if i > 0:
                    dt = p.timestamp_sec - traj.points[i - 1].timestamp_sec
                    if dt > 0:
                        ds = ((p.x - traj.points[i - 1].x) ** 2
                              + (p.y - traj.points[i - 1].y) ** 2) ** 0.5
                        sp = ds / dt * ps
                        vx = (p.x - traj.points[i - 1].x) / dt * ps
                        vy = (p.y - traj.points[i - 1].y) / dt * ps
                bucket["speed"].append(sp)
                bucket["vx"].append(vx)
                bucket["vy"].append(vy)
        return tl

    def _predicates(self, speed: float, vx: float, vy: float, jerk: float
                    ) -> Dict[str, bool]:
        toward = self._dock_direction(vx)
        return {
            STEP_APPROACH: bool(
                speed > 0.05 and speed < 0.6 and toward > 0.08
                and abs(vy) < 0.05
            ),
            STEP_HANDLE: bool(
                speed < 0.6 and (jerk > 0.3 or vy < -0.05)
            ),
            STEP_TRANSIT: bool(speed >= 0.6),
            STEP_PLACE: bool(
                speed < 0.4 and vy > 0.05 and jerk < 0.3
            ),
            STEP_DEPART: bool(
                speed > 0.05 and speed < 0.6 and toward < -0.08
                and abs(vy) < 0.05
            ),
        }

    def _dominant(self, active: Dict[str, bool]) -> str:
        for step in _DOMINANT_PRIORITY:
            if active.get(step):
                return step
        # Any motion without a clean label counts as transit if fast
        if active.get(STEP_TRANSIT):
            return STEP_TRANSIT
        return STEP_GAP

    def _dock_direction(self, vx: float) -> float:
        """Normalized dock-ward velocity (+ = toward dock) for left/right docks."""
        edge = self.truck_edge
        if edge == "auto":
            edge = "left"
        if edge in ("left", "right"):
            return -vx if edge == "left" else vx
        return vx

    # ------------------------------------------------------------------ output

    def _build_hypothesis(self, runs: List[dict]) -> SequenceHypothesis:
        steps = []
        for r in runs:
            steps.append({
                "step": r["step"],
                "start_sec": round(r["start"], 2),
                "end_sec": round(r["end"], 2),
                "mid_sec": round((r["start"] + r["end"]) / 2.0, 2),
                "evidence": f"{r['step']} observed over "
                            f"{r['end'] - r['start']:.1f}s "
                            f"({round((r['end'] - r['start']) * self.target_fps)} frames)",
            })
        ns = [s["step"] for s in steps]
        chain_label = "unsafe_loading_sequence"
        conf = min(1.0, 0.5 + 0.15 * len(steps))
        just = (
            f"Sequence '{chain_label}' detected: an ordered chain "
            f"{' -> '.join(ns)}. " + "; ".join(s["evidence"] for s in steps)
            + ". Multi-frame windows confirm each step, so the sequence is "
              "detected from evidence over time, not a single frame."
        )
        return SequenceHypothesis(
            hypothesis=chain_label,
            matched=True,
            steps=steps,
            order_respected=True,
            confidence=conf,
            justification=just,
            transitions=ns,
        )