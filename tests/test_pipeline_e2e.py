# tests/test_pipeline_e2e.py
"""
End-to-end smoke test: run process_clip on a real clip and verify that
motion detection → tracking → trajectory → behaviour match → physics →
risk scoring produce at least one RiskEvent with all required fields.

Skips automatically when data/clips/ is empty (CI, fresh clone).
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import pytest

# Ensure repo root is importable
_repo = pathlib.Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

CLIPS_DIR = _repo / "data" / "clips"
REAL_CLIPS = sorted(CLIPS_DIR.glob("*.mp4")) if CLIPS_DIR.exists() else []


@pytest.fixture(scope="module")
def sample_clip() -> pathlib.Path:
    if not REAL_CLIPS:
        pytest.skip("data/clips/ is empty — no real clips to test against")
    # Use the shortest clip to keep the test fast
    return min(REAL_CLIPS, key=lambda p: p.stat().st_size)


@pytest.mark.skipif(not REAL_CLIPS, reason="no real clips available")
def test_pipeline_produces_events(sample_clip: pathlib.Path) -> None:
    """Run the full pipeline on a real clip (capped at 120 frames ≈ 30 s)."""
    from pipeline.runner import process_clip

    events, trajectories = process_clip(str(sample_clip), max_frames=120)

    assert len(events) > 0, "Expected at least one RiskEvent from the clip"

    for ev in events:
        # Every event must have a non-empty justification
        assert ev.justification, f"Event {ev.event_id} has no justification"
        assert isinstance(ev.justification, str)

        # Risk score must be a finite number in [0, 1]
        assert 0.0 <= ev.risk_score <= 1.0, (
            f"Risk score {ev.risk_score} out of range for event {ev.event_id}"
        )

        # Risk level must be one of the four canonical levels
        assert ev.risk_level in ("low", "medium", "high", "critical"), (
            f"Invalid risk_level '{ev.risk_level}' for event {ev.event_id}"
        )

        # Behaviour class must be non-empty
        assert ev.behavior_class, f"Event {ev.event_id} has no behavior_class"

        # Event ID must be a non-empty string
        assert ev.event_id and isinstance(ev.event_id, str)

        # Timestamp must be positive
        assert ev.timestamp_sec > 0, (
            f"Event {ev.event_id} has non-positive timestamp {ev.timestamp_sec}"
        )

        # Physics severity must be in [0, 1]
        assert 0.0 <= ev.physics.severity_score <= 1.0

    print(f"\n  ✓ {sample_clip.name}: {len(events)} event(s), "
          f"{len(trajectories)} trajectory/trajectories")


@pytest.mark.skipif(not REAL_CLIPS, reason="no real clips available")
def test_trajectories_have_points(sample_clip: pathlib.Path) -> None:
    """Trajectories must contain at least one point each."""
    from pipeline.runner import process_clip

    _, trajectories = process_clip(str(sample_clip), max_frames=120)

    assert len(trajectories) > 0, "Expected at least one trajectory"
    for t in trajectories:
        assert len(t.points) > 0, (
            f"Trajectory {t.track_id} has no points"
        )
