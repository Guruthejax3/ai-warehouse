"""Unit tests for replay generator, assistant event store, and voice coach."""

from __future__ import annotations

import json

import pytest

from pipeline.assistant.rag import EventStore
from pipeline.replay.generator import build_replay, replay_to_dict
from pipeline.types import (
    BehaviorMatch,
    PhysicsState,
    RiskEvent,
    Trajectory,
    TrajectoryPoint,
)
from pipeline.voice.coach import VoiceCoach


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _event(event_id="e1", level="high", score=0.6, cls="product_dropped"):
    return RiskEvent(
        event_id=event_id,
        timestamp_sec=4.0,
        frame_idx=20,
        source_video="clip.mp4",
        behavior_class=cls,
        risk_score=score,
        risk_level=level,
        dtw_match=BehaviorMatch(cls, 1.2, 0.7, "ex_1", "matched"),
        physics=PhysicsState(1, physics_valid=True, severity_score=0.6),
        justification="dropped from height",
    )


@pytest.fixture()
def store(tmp_path):
    return EventStore(db_url=None, sqlite_path=str(tmp_path / "events.db"))


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _traj(n=20):
    pts = [
        TrajectoryPoint(
            frame_idx=i, timestamp_sec=i / 5.0, x=i * 0.1, y=3.0 + 0.1 * i
        )
        for i in range(n)
    ]
    return Trajectory(track_id=1, points=pts)


def test_replay_normalized_path():
    replay = build_replay(_event(), trajectory=_traj())
    d = replay_to_dict(replay)
    # Normalized canvas space: coordinates in [0, 1]
    for x, y in d["actual_path"]:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
    assert d["event_id"] == "e1"


def test_replay_correct_path_aligned():
    correct = [{"x": 0.0, "y": 3.0, "timestamp_sec": i / 5.0} for i in range(20)]
    replay = build_replay(_event(), trajectory=_traj(), correct_points=correct)
    d = replay_to_dict(replay)
    assert d["correct_path"]
    assert d["duration_sec"] > 0


def test_replay_single_point_no_divzero():
    one = Trajectory(track_id=1, points=[TrajectoryPoint(1, 0.2, 5.0, 5.0)])
    replay = build_replay(_event(), trajectory=one)
    d = replay_to_dict(replay)
    assert d["actual_path"]  # degrades to centre dot, no exception


# ---------------------------------------------------------------------------
# Assistant event store (SQLite)
# ---------------------------------------------------------------------------

def _insert_base(store, event_id, cls, score, level, zone="bay_7"):
    store.insert_event({
        "event_id": event_id,
        "created_at": "2026-09-05T10:00:00+00:00",
        "timestamp_sec": 12.0,
        "frame_idx": 60,
        "source_video": "clip.mp4",
        "behavior_class": cls,
        "risk_score": score,
        "risk_level": level,
        "physics_valid": True,
        "physics_severity": 0.5,
        "dtw_distance": 1.0,
        "justification": f"{cls} event",
        "zone_id": zone,
    })


def test_store_query_high_risk_level_ranks(store):
    _insert_base(store, "e1", "material_pushed_or_thrown", 0.8, "critical")
    _insert_base(store, "e2", "product_dragged", 0.3, "medium")
    # 'medium' must include critical + high events despite lexicographic order
    mid = store.query_high_risk(7, "medium")
    crit = store.query_high_risk(7, "critical")
    assert {e["event_id"] for e in mid} == {"e1", "e2"}
    assert {e["event_id"] for e in crit} == {"e1"}


def test_store_behavior_stats(store):
    _insert_base(store, "e1", "product_dropped", 0.8, "critical")
    _insert_base(store, "e2", "product_dropped", 0.4, "medium")
    stats = store.behavior_stats("product_dropped", 7)
    assert stats[0]["n"] == 2
    assert abs(stats[0]["avg_risk"] - 0.6) < 0.001


def test_store_bay_and_detail(store):
    _insert_base(store, "e1", "product_dropped", 0.8, "critical", zone="bay_2")
    assert store.bay_risk("bay_2")[0]["event_id"] == "e1"
    detail = store.event_detail("e1")
    assert detail["behavior_class"] == "product_dropped"
    assert store.event_detail("nope") is None


# ---------------------------------------------------------------------------
# Voice coach
# ---------------------------------------------------------------------------

def test_voice_message_high_risk():
    vc = VoiceCoach(enabled=False)
    msg = vc.build_message(_event())
    assert "High risk behaviour" in msg


def test_voice_disabled_logs_returns_false():
    vc = VoiceCoach(enabled=False)
    assert vc.speak("test") is False  # never blocks, never raises


def test_voice_gated_message():
    ev = _event(level="critical")
    ev.metadata = {"gated_by_physics": True}
    vc = VoiceCoach(enabled=False)
    msg = vc.build_message(ev)
    assert "no alert" in msg.lower()