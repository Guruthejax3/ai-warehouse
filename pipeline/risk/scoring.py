"""Risk scoring — the decision engine of the pipeline.

Prescribed formula:
    risk_score = w1*(1 - dtw_distance_normalized)
               + w2*physics_severity
               + w3*fragility_class
               + w4*repetition_count
               + w5*zone_criticality

Levels: low (<0.25), medium (<0.50), high (<0.75), critical (>=0.75).

Every event MUST carry a human-readable justification. The physics cross-check
is enforced before an alert is emitted (``physics.mandatory_before_alert``).
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

import numpy as np

from pipeline.types import (
    BehaviorMatch,
    PhysicsState,
    RiskEvent,
    RiskLevel,
)

logger = logging.getLogger(__name__)

_LEVELS = [("low", 0.25), ("medium", 0.50), ("high", 0.75), ("critical", 1.001)]


class RiskScorer:
    """Compute risk scores and produce explainable RiskEvents.

    Args:
        weights: dict with keys w1_dtw_match, w2_physics_severity,
                 w3_fragility, w4_repetition, w5_zone_criticality.
        fragility_classes: mapping of fragility label -> 0-1 difficulty.
        repetition_weight_bonus: how much each repeated event adds.
        mandatory_before_alert: refuse to emit alerts when physics cross-check
            fails (default True).
        evidence_trim_sec: how many seconds each side of the event peak are
            kept for the evidence clip.
        target_fps: processing FPS (used to convert seconds to frames).
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        fragility_classes: Optional[Dict[str, float]] = None,
        repetition_weight_bonus: float = 0.05,
        mandatory_before_alert: bool = True,
        evidence_trim_sec: int = 5,
        target_fps: float = 5.0,
    ) -> None:
        self.weights = weights or {
            "w1_dtw_match": 0.25,
            "w2_physics_severity": 0.25,
            "w3_fragility": 0.15,
            "w4_repetition": 0.20,
            "w5_zone_criticality": 0.15,
        }
        self.fragility_classes = fragility_classes or {
            "standard": 0.3,
            "fragile": 0.7,
            "very_fragile": 1.0,
        }
        self.repetition_weight_bonus = repetition_weight_bonus
        self.mandatory_before_alert = mandatory_before_alert
        self.evidence_trim_sec = evidence_trim_sec
        self.target_fps = target_fps
        self._event_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------ public

    def reset_counts(self) -> None:
        """Reset repetition counters (start of new video)."""
        self._event_counts = {}

    def score_event(
        self,
        source_video: str,
        frame_idx: int,
        timestamp_sec: float,
        behavior: BehaviorMatch,
        physics: PhysicsState,
        physics_ok: bool = True,
        physics_reasons: Optional[List[str]] = None,
        fragility: str = "standard",
        zone_criticality: float = 0.3,
        behavior_override: Optional[str] = None,
        rule_reason: str = "",
    ) -> RiskEvent:
        """Build a fully-justified RiskEvent for one trajectory.

        The physics cross-check is the mandatory gate: if it fails and
        mandatory_before_alert is True, no alert is emitted (risk scored but
        ``risk_level`` set to "low" for logging only).
        """
        self._event_counts[behavior.behavior_class] = (
            self._event_counts.get(behavior.behavior_class, 0) + 1
        )
        repetition_count = self._event_counts[behavior.behavior_class]

        behavior_class = behavior_override or behavior.behavior_class

        # w1: match quality — normalized distance to 0-1
        dtw_norm = self._normalized_dtw(behavior.dtw_distance)
        w1 = self.weights.get("w1_dtw_match", 0.25) * (1.0 - dtw_norm)

        # w2: physics severity
        w2 = self.weights.get("w2_physics_severity", 0.25) * physics.severity_score

        # w3: fragility of the product
        fragility_val = self.fragility_classes.get(fragility, 0.3)
        w3 = self.weights.get("w3_fragility", 0.15) * fragility_val

        # w4: repetition — repeated offences escalate
        repeat_norm = min(1.0, repetition_count / 5.0)
        w4 = self.weights.get("w4_repetition", 0.20) * repeat_norm

        # w5: zone criticality (configurable per-bay)
        w5 = self.weights.get("w5_zone_criticality", 0.15) * zone_criticality

        raw = w1 + w2 + w3 + w4 + w5
        risk_score = float(np.clip(raw, 0.0, 1.0))

        # Mandatory physics gate
        gated = False
        if self.mandatory_before_alert and not physics_ok:
            risk_score = float(np.clip(risk_score, 0.0, 0.24))  # cap strictly under Low threshold
            gated = True

        risk_level = self._classify(risk_score)

        # Evidence trim (frames) around the peak frame
        ev_start, ev_end = self._evidence_window(frame_idx)

        justification = self._build_justification(
            behavior_class=behavior_class,
            risk_score=risk_score,
            risk_level=risk_level,
            behavior=behavior,
            physics=physics,
            fragility=fragility,
            repetition_count=repetition_count,
            zone_criticality=zone_criticality,
            gated=gated,
            physics_reasons=physics_reasons,
            rule_reason=rule_reason,
        )

        return RiskEvent(
            event_id=uuid.uuid4().hex[:12],
            timestamp_sec=float(timestamp_sec),
            frame_idx=int(frame_idx),
            source_video=source_video,
            behavior_class=behavior_class,
            risk_score=risk_score,
            risk_level=risk_level,
            dtw_match=behavior,
            physics=physics if physics_ok else physics,
            justification=justification,
            evidence_clip_start=int(ev_start),
            evidence_clip_end=int(ev_end),
            trajectory_id=physics.track_id,
            metadata={
                "gated_by_physics": gated,
                "repetition_count": repetition_count,
                "fragility": fragility,
                "zone_criticality": zone_criticality,
                "dtw_norm": dtw_norm,
                "w1": w1,
                "w2": w2,
                "w3": w3,
                "w4": w4,
                "w5": w5,
            },
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _normalized_dtw(dtw_distance: float) -> float:
        """Map DTW distance to 0-1 where 1 = far (bad match)."""
        return float(np.clip(dtw_distance / 6.0, 0.0, 1.0))

    def _classify(self, score: float) -> RiskLevel:
        if score < 0.25:
            return "low"
        if score < 0.50:
            return "medium"
        if score < 0.75:
            return "high"
        return "critical"

    def _evidence_window(self, frame_idx: int) -> tuple:
        half = int(self.evidence_trim_sec * self.target_fps)
        return max(0, frame_idx - half), frame_idx + half

    def _build_justification(
        self,
        behavior_class: str,
        risk_score: float,
        risk_level: RiskLevel,
        behavior: BehaviorMatch,
        physics: PhysicsState,
        fragility: str,
        repetition_count: int,
        zone_criticality: float,
        gated: bool,
        physics_reasons: Optional[List[str]],
        rule_reason: str = "",
    ) -> str:
        """Compose a human-readable explanation, never a bare number."""
        parts: List[str] = []
        if rule_reason:
            # Deterministic-rule event (zone placement, stacking, FSM, ...):
            # explain the rule that fired rather than implying a DTW match.
            parts.append(f"Rule fired: {rule_reason}")
            parts.append(
                f"Behavior '{behavior_class}' confirmed by a deterministic "
                f"rule (no exemplar matching)."
            )
        else:
            parts.append(
                f"Behavior '{behavior_class}' matched with "
                f"{behavior.confidence:.0%} confidence (DTW distance "
                f"{behavior.dtw_distance:.2f})."
            )
        parts.append(
            f"Physics: {physics.justification.strip()}"
        )
        if physics_reasons and not gated:
            parts.append("Kinematic confirmation: " + "; ".join(physics_reasons))
        parts.append(
            f"Product fragility class '{fragility}' (weight "
            f"{self.fragility_classes.get(fragility, 0.3):.2f})."
        )
        if repetition_count > 1:
            parts.append(
                f"Repeated event #{repetition_count} for this behavior — "
                f"escalating severity."
            )
        parts.append(f"Zone criticality {zone_criticality:.2f}.")
        if gated:
            parts.append(
                "ALERT BLOCKED: physics cross-check failed; recorded for "
                "review only (score capped at Low)."
            )
        parts.append(
            f"Overall risk {risk_score:.2f} -> {risk_level.upper()}."
        )
        return " ".join(parts)