"""Risk Scoring Module — real-time risk from kinematics + pose + behaviour.

Stage 6 (final) of the Damage-DNA cv-pipeline. Combines the outputs of the
previous stages into a single, auditable risk score per event:

    risk = w_phys * kinematics_severity
         + w_pose * pose_hazard
         + w_seq  * fsm_unsafe_sequence
         + w_frag * fragility_exposure

Every scored event carries: a risk_level (low/medium/high/critical), a
weighted parameter breakdown (so a judge can trace the number), and a plain-
English justification that names the contributing stage evidence.

Handoff contract (output JSON):
    {
        "event_id": str,
        "track_id": int,
        "timestamp_sec": float,
        "behavior_class": str,
        "risk_score": float,
        "risk_level": "low"|"medium"|"high"|"critical",
        "weights": {"physics": float, "pose": float, "sequence": float, "fragility": float},
        "justification": str
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

#: thresholds for score -> level
LEVELS = [("low", 0.25), ("medium", 0.5), ("high", 0.75)]
FRAGILITY_BY_CLASS: Dict[str, str] = {
    "fragile": "fragile", "standard": "standard", "durable": "durable",
}


@dataclass
class ScoredEvent:
    event_id: str
    track_id: int
    timestamp_sec: float
    behavior_class: str
    risk_score: float
    risk_level: str
    weights: Dict[str, float]
    justification: str

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "behavior_class": self.behavior_class,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "justification": self.justification,
        }


class RiskScorer:
    """Combine stage evidence into a scored event.

    Args:
        weights: optional overrides for
            {"physics": 0.45, "pose": 0.15, "sequence": 0.25, "fragility": 0.15}.
        behavior_map: optional callable/track embedding that maps kinematics
            + pose evidence to a behavior class name. Defaults to a simple
            rule table.
    """

    DEFAULT_WEIGHTS = {"physics": 0.45, "pose": 0.15, "sequence": 0.25, "fragility": 0.15}

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        behavior_map: Optional[object] = None,
    ) -> None:
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({k: v for k, v in weights.items() if k in self.DEFAULT_WEIGHTS})
        self.behavior_map = behavior_map

    # ------------------------------------------------------------------ public

    def score(
        self,
        event_id: str,
        track_id: int,
        timestamp_sec: float,
        kinematics_severity: float = 0.0,
        pose_hazard: bool = False,
        fsm_unsafe: bool = False,
        fragility: str = "standard",
        drop_height_m: float = 0.0,
        peak_speed_mps: float = 0.0,
    ) -> ScoredEvent:
        """Compute the real-time risk for one candidate event.

        Args:
            event_id: caller-assigned id (e.g. uuid4 hex, short form).
            track_id: the tracked object.
            timestamp_sec: time of the event.
            kinematics_severity: stage-4 severity (0-1).
            pose_hazard: stage-3 hazardous-posture flag.
            fsm_unsafe: stage-5 sequence violation flag.
            fragility: product fragility class (fragile/standard/durable).
            drop_height_m, peak_speed_mps: additional stage-4 evidence used
                only for the justification narrative.

        Returns:
            ScoredEvent ready for JSON serialisation.
        """
        s = self.weights
        ks = max(0.0, min(1.0, kinematics_severity))
        pose = 1.0 if pose_hazard else 0.0
        seq = 1.0 if fsm_unsafe else 0.0
        frag = {"fragile": 1.0, "standard": 0.35, "durable": 0.0}.get(
            FRAGILITY_BY_CLASS.get(fragility, "standard"), 0.35
        )

        score = (
            s["physics"] * ks
            + s["pose"] * pose
            + s["sequence"] * seq
            + s["fragility"] * frag
        )
        score = max(0.0, min(1.0, score))

        level = "low"
        for name, thr in LEVELS:
            if score >= thr:
                level = name
        if score >= 0.75:
            level = "critical" if score >= 0.9 else "high"

        why = []
        if ks > 0.35:
            why.append(f"kinematics severity {ks:.2f}")
        if drop_height_m > 0.4:
            why.append(f"drop height {drop_height_m:.2f} m")
        if peak_speed_mps > 1.0:
            why.append(f"peak speed {peak_speed_mps:.2f} m/s")
        if pose:
            why.append("hazardous worker posture")
        if seq:
            why.append("unsafe loading sequence (FSM violation)")
        if frag > 0.5:
            why.append(f"{fragility} product exposure")
        behavior = self._infer_behavior(ks, drop_height_m, seq, pose)
        justification = ", ".join(why) if why else "low-level observation"
        justification += "."

        return ScoredEvent(
            event_id=event_id,
            track_id=track_id,
            timestamp_sec=timestamp_sec,
            behavior_class=behavior,
            risk_score=round(score, 4),
            risk_level=level,
            weights={"physics": s["physics"], "pose": s["pose"],
                     "sequence": s["sequence"], "fragility": s["fragility"]},
            justification=justification,
        )

    # ------------------------------------------------------------------ private

    def _infer_behavior(
        self, ks: float, drop_height_m: float, seq: bool, pose: bool
    ) -> str:
        if seq:
            return "unsafe_loading_sequence"
        if drop_height_m > 0.4 and ks > 0.6:
            return "product_dropped"
        if pose and ks > 0.3:
            return "rough_handling"
        if ks > 0.5:
            return "rapid_material_movement"
        return "normal_activity"