"""Unit tests for risk scoring: formula, levels, and explainable justification."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.risk.scoring import RiskScorer
from pipeline.types import BehaviorMatch, PhysicsState


def _behavior(cls="product_dropped", dtw=1.0, conf=0.8):
    return BehaviorMatch(
        behavior_class=cls,
        dtw_distance=dtw,
        confidence=conf,
        exemplar_id="ex_1",
        justification=f"matched {cls}",
    )


def _physics(severity=0.6, valid=True):
    return PhysicsState(
        track_id=1,
        severity_score=severity,
        physics_valid=valid,
        justification="measured drop 0.8 m, impact 3.0 m/s",
    )


def test_low_level():
    scorer = RiskScorer()
    event = scorer.score_event(
        source_video="v.mp4",
        frame_idx=100,
        timestamp_sec=20.0,
        behavior=_behavior(dtw=5.0, conf=0.1),
        physics=_physics(severity=0.1),
        fragility="standard",
        zone_criticality=0.0,
    )
    assert event.risk_level == "low"
    assert event.risk_score < 0.25


def test_critical_level():
    scorer = RiskScorer()
    event = scorer.score_event(
        source_video="v.mp4",
        frame_idx=100,
        timestamp_sec=20.0,
        behavior=_behavior(dtw=0.2, conf=1.0),
        physics=_physics(severity=1.0),
        fragility="very_fragile",
        zone_criticality=1.0,
    )
    assert event.risk_level == "critical"
    assert event.risk_score >= 0.75


def test_justification_always_present():
    scorer = RiskScorer()
    event = scorer.score_event(
        source_video="v.mp4",
        frame_idx=100,
        timestamp_sec=20.0,
        behavior=_behavior(),
        physics=_physics(),
    )
    assert event.justification
    # Explainability contract: the human-readable reason must mention the
    # behaviour and the overall level, not just a bare number.
    assert "Behavior" in event.justification
    assert event.risk_level.upper() in event.justification


def test_physics_gate_blocks_alert():
    """If physics cross-check fails and mandatory_before_alert is True, the
    event is capped to Low and flagged for review — never emitted as an alert."""
    scorer = RiskScorer(mandatory_before_alert=True)
    event = scorer.score_event(
        source_video="v.mp4",
        frame_idx=100,
        timestamp_sec=20.0,
        behavior=_behavior(dtw=0.3, conf=1.0),
        physics=_physics(severity=1.0),
        physics_ok=False,
        physics_reasons=["impact speed too low"],
    )
    assert event.risk_level == "low"
    assert event.metadata["gated_by_physics"] is True
    assert "ALERT BLOCKED" in event.justification


def test_physics_gate_off_when_disabled():
    scorer = RiskScorer(mandatory_before_alert=False)
    event = scorer.score_event(
        source_video="v.mp4",
        frame_idx=100,
        timestamp_sec=20.0,
        behavior=_behavior(dtw=0.0, conf=1.0),
        physics=_physics(severity=1.0),
        physics_ok=False,
        fragility="very_fragile",
        zone_criticality=1.0,
    )
    assert event.metadata["gated_by_physics"] is False
    assert event.risk_level == "critical"


def test_repetition_escalates():
    scorer = RiskScorer()
    a = scorer.score_event(
        source_video="v.mp4", frame_idx=1, timestamp_sec=1.0,
        behavior=_behavior(), physics=_physics(),
    )
    b = scorer.score_event(
        source_video="v.mp4", frame_idx=10, timestamp_sec=2.0,
        behavior=_behavior(), physics=_physics(),
    )
    assert b.risk_score > a.risk_score
    assert b.metadata["repetition_count"] == 2


def test_score_bounded_zero_to_one():
    scorer = RiskScorer()
    event = scorer.score_event(
        source_video="v.mp4", frame_idx=1, timestamp_sec=1.0,
        behavior=_behavior(), physics=_physics(),
    )
    assert 0.0 <= event.risk_score <= 1.0


def test_reset_counts():
    scorer = RiskScorer()
    scorer.score_event(
        source_video="v.mp4", frame_idx=1, timestamp_sec=1.0,
        behavior=_behavior(), physics=_physics(),
    )
    scorer.reset_counts()
    event = scorer.score_event(
        source_video="v.mp4", frame_idx=1, timestamp_sec=1.0,
        behavior=_behavior(), physics=_physics(),
    )
    assert event.metadata["repetition_count"] == 1


def test_evidence_window():
    scorer = RiskScorer(evidence_trim_sec=5, target_fps=5.0)
    event = scorer.score_event(
        source_video="v.mp4", frame_idx=100, timestamp_sec=20.0,
        behavior=_behavior(), physics=_physics(),
    )
    # ±5 s at 5 fps = ±25 frames
    assert event.evidence_clip_end - event.evidence_clip_start == 50
