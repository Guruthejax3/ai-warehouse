"""Backend tests: FastAPI endpoints + thread-safe WS broadcast drain.

Sets an isolated SQLite DB before importing the backend so tests never touch
data/backend.db. Ingest is NOT exercised here (a real clip is slow); the
endpoints are tested against a seeded event.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time

# Isolate the DB *before* backend imports wire up the module engine.
_tmp = tempfile.mkdtemp(prefix="rtw_test_")
os.environ["REPLAYTWIN_DB_SQLITE"] = os.path.join(_tmp, "test_backend.db")
os.environ["DATABASE_URL"] = ""  # force the SQLite fallback in tests
os.environ["ANTHROPIC_API_KEY"] = ""  # assistant degrades to "not configured"

from fastapi.testclient import TestClient  # noqa: E402

from backend import db, models  # noqa: E402
from backend.main import _drain_broadcasts, _publish, app  # noqa: E402


def _seed_event() -> str:
    """Insert a fake event + trajectory, return its human-facing event_id.

    Idempotent: drops any prior seed rows so multiple tests can seed safely.
    """
    session = db.SessionLocal()
    try:
        prior = session.query(models.Event).filter(models.Event.event_id == "seed1").all()
        for p in prior:
            session.delete(p)
        session.flush()
        row = models.Event(
            event_id="seed1",
            timestamp_sec=10.5,
            frame_idx=42,
            source_video="sampler.mp4",
            behavior_class="product_dropped",
            risk_score=0.72,
            risk_level="high",
            physics_valid=True,
            physics_severity=0.6,
            justification="dropped carton from height (drop-height 1.4 m)",
            zone_id="bay_7",
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        session.add(
            models.TrajectoryModel(
                event_id=row.id,
                track_id=3,
                class_label="object",
                points=[
                    {"frame_idx": i, "timestamp_sec": i / 5.0, "x": 0.1 * i, "y": 3.0, "z_est": 0.0}
                    for i in range(10)
                ],
            )
        )
        session.commit()
    finally:
        session.close()
    return "seed1"


def test_health() -> None:
    with TestClient(app) as c:
        r = c.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_events_and_detail_and_replay() -> None:
    _seed_event()
    with TestClient(app) as c:
        rows = c.get("/api/events").json()
        assert any(r["event_id"] == "seed1" for r in rows)

        detail = c.get("/api/events/seed1").json()
        assert detail["event"]["risk_level"] == "high"
        assert "dropped" in detail["event"]["justification"]

        replay = c.get("/api/replay/seed1").json()
        assert len(replay["actual_path"]) == 10  # persisted trajectory


def test_behaviors_and_voice() -> None:
    with TestClient(app) as c:
        classes = c.get("/api/behaviors").json()["behavior_classes"]
        assert "product_dropped" in classes

        v = c.post(
            "/api/voice/generate",
            json={"behavior_class": "product_dropped", "risk_level": "critical"},
        ).json()
        assert "high risk" in v["message"].lower()
        # Multilingual voice coach renders the class as a natural-language
        # label, not the raw snake_case id.
        assert "a product was dropped" in v["message"].lower()
        assert "product_dropped" not in v["message"]


def test_assistant_no_key_degrades_helpfully() -> None:
    with TestClient(app) as c:
        r = c.post("/api/assistant", json={"message": "how many critical events?"})
        assert r.status_code == 200
        assert "ANTHROPIC_API_KEY" in r.json()["message"]


def test_feedback_resolves_human_event_id() -> None:
    _seed_event()
    with TestClient(app) as c:
        r = c.post(
            "/api/feedback",
            json={"event_id": "seed1", "annotator": "tester", "correct_risk_level": "critical"},
        )
        assert r.status_code == 200
        assert r.json()["ok"]


def test_feedback_unknown_event_404() -> None:
    with TestClient(app) as c:
        r = c.post("/api/feedback", json={"event_id": "does-not-exist"})
        assert r.status_code == 404


def test_broadcast_drain_delivers_across_threads() -> None:
    """_publish() from a worker thread reaches the drain task on the loop."""
    delivered = []
    original = type(app).__name__  # noqa: F841

    async def fake_broadcast(payload):
        delivered.append(payload)
        return None

    # Patch the manager's broadcast on the app object for this test.
    import backend.main as bm

    orig = bm.manager.broadcast
    bm.manager.broadcast = fake_broadcast

    async def _run() -> None:
        bm._broadcast_loop = asyncio.get_running_loop()
        bm._broadcast_queue = asyncio.Queue()
        task = asyncio.create_task(_drain_broadcasts())
        ev = {"event_id": "br1", "behavior_class": "x"}

        def worker() -> None:
            time.sleep(0.1)
            _publish({"type": "event", "data": ev})

        t = threading.Thread(target=worker)
        t.start()
        await asyncio.sleep(0.5)
        task.cancel()

    try:
        asyncio.run(_run())
    finally:
        bm.manager.broadcast = orig

    assert delivered == [{"type": "event", "data": {"event_id": "br1", "behavior_class": "x"}}]