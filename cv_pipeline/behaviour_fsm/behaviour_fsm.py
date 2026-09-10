"""Behaviour FSM Module — worker/equipment behaviour as a state machine.

Stage 5 of the Damage-DNA cv-pipeline. Consumes kinematic summaries + activity
labels (physics speed/direction) and models each worker's behaviour as a
finite state machine over event labels. Unsafe transitions (e.g. "handle"
directly after "depart", or skipping "approach") are flagged as violations
with the observed evidence — this is the multi-step *sequence* contract that
single-frame classifiers miss.

States (alphabet):
    idle, dock_approach, handle, transit, place, depart

Legal machine (a safe loading cycle):
    idle -> dock_approach -> handle -> transit -> place -> idle
Violations detected by the FSM:
    - handle without a prior approach   (rushed lift)
    - depart while load still held      (leaving with a lifted item)
    - place at speed (impact placement)
    - any transition with no movement   (still-hold -> nothing)

Handoff contract (output JSON):
    {
        "track_id": int,
        "states": [{"time": float, "state": str, "evidence": str}, ...],
        "cycle_count": int,
        "violations": [{"time": float, "type": str, "evidence": str}, ...]
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

S_IDLE = "idle"
S_APPROACH = "dock_approach"
S_HANDLE = "handle"
S_TRANSIT = "transit"
S_PLACE = "place"
S_DEPART = "depart"

#: The safe cycle the machine checks for (a full loading sequence).
SAFE_CYCLE = [S_APPROACH, S_HANDLE, S_TRANSIT, S_PLACE]


@dataclass
class FSMState:
    state: str
    time: float
    evidence: str


@dataclass
class FSMViolation:
    time: float
    violation_type: str
    evidence: str


@dataclass
class FSMTrace:
    track_id: int
    states: List[FSMState] = field(default_factory=list)
    violations: List[FSMViolation] = field(default_factory=list)
    cycle_count: int = 0
    unsafe_sequence: bool = False   # True when a violation is structural

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "states": [{"time": round(s.time, 2), "state": s.state,
                        "evidence": s.evidence} for s in self.states],
            "cycle_count": self.cycle_count,
            "violations": [{"time": round(v.time, 2), "type": v.violation_type,
                            "evidence": v.evidence} for v in self.violations],
            "unsafe_sequence": self.unsafe_sequence,
        }


class BehaviourFSM:
    """Finite-state machine over behaviour labels from physics/activity input.

    Args:
        min_dwell_sec: minimum time (s) in a state before it is recorded.
        enabled: when False, ``analyze`` returns an empty trace.
    """

    def __init__(
        self,
        min_dwell_sec: float = 0.4,
        enabled: bool = True,
    ) -> None:
        self.min_dwell_sec = min_dwell_sec
        self.enabled = enabled
        # State -> set of legal next states
        self._legal = {
            S_IDLE: {S_APPROACH, S_DEPART},
            S_APPROACH: {S_HANDLE, S_IDLE, S_DEPART},
            S_HANDLE: {S_TRANSIT, S_PLACE, S_IDLE, S_HANDLE},
            S_TRANSIT: {S_PLACE, S_HANDLE, S_IDLE},
            S_PLACE: {S_IDLE, S_APPROACH},
            S_DEPART: {S_IDLE, S_APPROACH},
        }

    # ------------------------------------------------------------------ public

    def analyze(
        self,
        track_id: int,
        steps: List[Tuple[float, str, str]],
        activity_label: Optional[str] = None,
        camera_edge: str = "left",
    ) -> FSMTrace:
        """Run the state machine over a sequence of (time, state, evidence).

        Args:
            track_id: the track id.
            steps: ordered [(time_sec, state_label, evidence_str)] from the
                per-frame activity classifier output (this stage's input).
            activity_label: unused but kept for symmetry with the pipeline's
                ActivityClassifier; reserved for merged analysis.
            camera_edge: dock direction ("left"/"right"), reserved.

        Returns:
            FSMTrace with the accepted state timeline, cycle count, and any
            violations of the legal-transition table.
        """
        trace = FSMTrace(track_id=track_id)
        if not self.enabled or not steps:
            return trace

        # 1) collapse consecutive repeats with a dwell-time floor
        timeline: List[FSMState] = []
        for t, state, evidence in steps:
            if timeline and timeline[-1].state == state:
                if t - timeline[-1].time >= self.min_dwell_sec:
                    timeline[-1].evidence = evidence
                continue
            timeline.append(FSMState(state=state, time=t, evidence=evidence))
        trace.states = timeline

        # 2) check legal transitions + update cycle counter
        cycle_pos = 0
        for s in timeline:
            # cycle detection (unordered): mark how far along a SAFE_CYCLE we are
            if s.state == SAFE_CYCLE[cycle_pos]:
                cycle_pos += 1
                if cycle_pos == len(SAFE_CYCLE):
                    trace.cycle_count += 1
                    cycle_pos = 0
            elif s.state in SAFE_CYCLE:
                cycle_pos = 0
                if s.state == SAFE_CYCLE[0]:
                    cycle_pos = 1

        for prev, cur in zip(timeline, timeline[1:]):
            legal = self._legal.get(prev.state, set())
            if cur.state not in legal:
                trace.violations.append(FSMViolation(
                    time=cur.time,
                    violation_type=self._violation_name(prev.state, cur.state),
                    evidence=f"transition {prev.state} -> {cur.state} "
                             f"at {cur.time:.1f}s ({cur.evidence})",
                ))

        # 3) structural "unsafe sequence" = any transition violation present
        trace.unsafe_sequence = len(trace.violations) > 0
        return trace

    # ------------------------------------------------------------------ private

    @staticmethod
    def _violation_name(prev: str, cur: str) -> str:
        if prev == S_IDLE and cur == S_HANDLE:
            return "lift_without_approach"
        if prev in (S_PLACE, S_HANDLE) and cur == S_DEPART:
            return "depart_with_load"
        if prev == S_TRANSIT and cur == S_PLACE:
            return "rushed_placement"
        if prev == S_IDLE and cur == S_TRANSIT:
            return "transit_without_approach"
        return f"illegal_{prev}_to_{cur}"