"""ReplayTwin FastAPI backend.

Endpoints:
    POST   /api/ingest         run pipeline on a clip, persist events
    GET    /api/events         query events with filters
    GET    /api/events/{id}    single event + replay payload
    GET    /api/replay/{id}    actual + correct-technique trajectories
    POST   /api/voice/generate TTS for a coaching message
    POST   /api/assistant      chat with the RAG assistant
    POST   /api/feedback       annotator feedback on an event
    WS     /ws/events          real-time event stream

Privacy contract: evidence clips are trimmed ±5 s and face-blurred; full raw
video is never stored or served.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend import models
from backend.db import get_db, init_db
from backend.schemas import (
    AssistantRequest,
    AssistantResponse,
    FeedbackRequest,
    IngestRequest,
    IngestResponse,
    VoiceRequest,
    VoiceResponse,
)
from pipeline.matching.dtw_matcher import ExemplarLibrary

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    global _broadcast_loop, _broadcast_queue
    init_db()
    _broadcast_loop = asyncio.get_running_loop()
    _broadcast_queue = asyncio.Queue()
    task = asyncio.create_task(_drain_broadcasts())
    yield
    task.cancel()
    _broadcast_loop = None


app = FastAPI(title="ReplayTwin", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo — tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

ROOT = Path(__file__).parent.parent
EVIDENCE_DIR = ROOT / "data" / "evidence_clips"


# ---------------------------------------------------------------------------
# WebSocket event hub
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, payload: dict) -> None:
        for ws in list(self.active):
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive / ping
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Ingest runs in a worker thread (run_in_threadpool) with no event loop, but
# broadcasts must be sent from the app's loop. `_publish` hands the payload to
# the loop via call_soon_threadsafe into an asyncio.Queue; a lifespan task
# drains that queue and fans each payload out to connected sockets. Nothing
# blocks a worker thread, so shutdown is clean.
_broadcast_loop: Optional[asyncio.AbstractEventLoop] = None
_broadcast_queue: Optional[asyncio.Queue] = None


async def _drain_broadcasts() -> None:
    """Loop: pop items off the queue and fan them out to connected sockets."""
    while True:
        payload = await _broadcast_queue.get()
        try:
            await manager.broadcast(payload)
        except Exception:
            logger.exception("broadcast failed")


def _publish(payload: dict) -> None:
    """Thread-safe broadcast from a sync context (ingest callback)."""
    loop = _broadcast_loop
    if loop is not None and _broadcast_queue is not None:
        loop.call_soon_threadsafe(_broadcast_queue.put_nowait, payload)


# ---------------------------------------------------------------------------
# Event persistence helpers
# ---------------------------------------------------------------------------

def _event_to_dict(ev) -> dict:
    md = getattr(ev, "metadata", {}) or {}
    physics = getattr(ev, "physics", None)
    dtw = getattr(ev, "dtw_match", None)
    return {
        "event_id": ev.event_id,
        "timestamp_sec": ev.timestamp_sec,
        "frame_idx": ev.frame_idx,
        "source_video": ev.source_video,
        "behavior_class": ev.behavior_class,
        "risk_score": ev.risk_score,
        "risk_level": ev.risk_level,
        "confidence": dtw.confidence if dtw else None,
        "dtw_distance": dtw.dtw_distance if dtw else None,
        "physics_valid": bool(physics and physics.physics_valid),
        "physics_severity": physics.severity_score if physics else None,
        "justification": ev.justification,
        "evidence_clip_start": ev.evidence_clip_start,
        "evidence_clip_end": ev.evidence_clip_end,
        "evidence_clip_path": getattr(ev, "evidence_clip_path", ""),
        "activity_type": getattr(ev, "activity_type", "unknown"),
        "zone_id": md.get("zone_id", "bay_0"),
        "metadata": md,
    }


def _persist_event(db: Session, ev, trajectory_by_track: Optional[dict] = None) -> models.Event:
    row = models.Event()
    md = getattr(ev, "metadata", {}) or {}
    physics = getattr(ev, "physics", None)
    dtw = getattr(ev, "dtw_match", None)
    row.event_id = ev.event_id
    row.timestamp_sec = ev.timestamp_sec
    row.frame_idx = ev.frame_idx
    row.source_video = ev.source_video
    row.behavior_class = ev.behavior_class
    row.risk_score = ev.risk_score
    row.risk_level = ev.risk_level
    row.confidence = dtw.confidence if dtw else None
    row.dtw_distance = dtw.dtw_distance if dtw else None
    row.physics_valid = bool(physics and physics.physics_valid)
    row.physics_severity = physics.severity_score if physics else None
    row.justification = ev.justification
    row.evidence_clip_start = ev.evidence_clip_start
    row.evidence_clip_end = ev.evidence_clip_end
    row.zone_id = md.get("zone_id", "bay_0")
    row.metadata_json = md
    db.add(row)
    db.commit()
    db.refresh(row)

    # Persist the trajectory that produced this event (replay payload source).
    traj = None
    if trajectory_by_track:
        traj = trajectory_by_track.get(ev.trajectory_id) or (
            trajectory_by_track.get(physics.track_id) if physics else None
        )
    if traj and traj.points:
        pts = [
            {
                "frame_idx": p.frame_idx,
                "timestamp_sec": p.timestamp_sec,
                "x": p.x,
                "y": p.y,
                "z_est": p.z_est,
            }
            for p in traj.points
        ]
        tm = models.TrajectoryModel(
            event_id=row.id,
            track_id=traj.track_id,
            class_label=traj.class_label or "object",
            points=pts,
            total_displacement=traj.total_displacement,
            mean_speed=traj.mean_speed,
            max_speed=traj.max_speed,
        )
        db.add(tm)
        db.commit()
    return row


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_video(req: IngestRequest, db: Session = Depends(get_db)) -> IngestResponse:
    """Run the pipeline on a clip and persist scored events.

    Clip paths are resolved relative to the repo data/clips dir (or absolute).
    Evidence clips are trimmed ±5 s and face-blurred; the raw video is never
    stored or served.
    """
    video = Path(req.video_path)
    if not video.is_absolute():
        video = ROOT / data_path(video)
    if not video.exists():
        raise HTTPException(404, f"Clip not found: {req.video_path}")

    # set params
    max_frames = req.max_frames or 0

    def _process():
        from pipeline.runner import process_clip

        events, trajectories = process_clip(video, max_frames=max_frames)

        # Persist + evidence clips
        persisted: List[str] = []
        trajectory_by_track = {t.track_id: t for t in trajectories}
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        for ev in events:
            row = _persist_event(db, ev, trajectory_by_track)
            persisted.append(ev.event_id)
            # Evidence clip (privacy-safe ±5 s, blurred)
            try:
                from pipeline.privacy.blur import extract_evidence_clip

                clip = extract_evidence_clip(
                    video,
                    peak_sec=ev.timestamp_sec,
                    output_dir=EVIDENCE_DIR,
                    trim_sec=float(cfg_trim()),
                    event_id=ev.event_id,
                )
                ev.evidence_clip_path = str(clip.path)
                row.evidence_clip_path = str(clip.path)
                db.commit()
            except Exception as exc:
                logger.warning("evidence clip for %s failed: %s", ev.event_id, exc)
            # broadcast live
            _publish({"type": "event", "data": _event_to_dict(ev)})
        return persisted, len(events)

    persisted, total = await run_in_threadpool(_process)
    return IngestResponse(video_path=req.video_path, events_persisted=persisted, total=total)


def data_path(video: Path) -> Path:
    from pipeline.config import load_config

    cfg = load_config()
    source = cfg["pipeline"].get("video_source", "data/clips/")
    return Path(source) / video


def cfg_trim() -> float:
    from pipeline.config import load_config

    return float(load_config()["risk"].get("evidence_trim_sec", 5.0))


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@app.get("/api/events")
async def list_events(
    behavior: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Query events with optional behavior_class / risk_level filters."""
    q = db.query(models.Event)
    if behavior:
        q = q.filter(models.Event.behavior_class == behavior)
    if level:
        q = q.filter(models.Event.risk_level == level)
    rows = q.order_by(models.Event.created_at.desc()).limit(min(limit, 500)).all()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(r: models.Event) -> dict:
    return {
        "event_id": r.event_id,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "timestamp_sec": r.timestamp_sec,
        "frame_idx": r.frame_idx,
        "source_video": r.source_video,
        "behavior_class": r.behavior_class,
        "risk_score": r.risk_score,
        "risk_level": r.risk_level,
        "confidence": r.confidence,
        "dtw_distance": r.dtw_distance,
        "physics_valid": r.physics_valid,
        "physics_severity": r.physics_severity,
        "justification": r.justification,
        "evidence_clip_path": r.evidence_clip_path,
        "zone_id": r.zone_id,
    }


@app.get("/api/events/{event_id}")
async def get_event(event_id: str, db: Session = Depends(get_db)):
    row = db.query(models.Event).filter(models.Event.event_id == event_id).first()
    if row is None:
        raise HTTPException(404, f"Event {event_id} not found")
    return {"event": _row_to_dict(row), "replay": _replay_for(row)}


@app.get("/api/replay/{event_id}")
async def get_replay(event_id: str):
    from backend.db import SessionLocal as _SL

    db = _SL()
    try:
        row = db.query(models.Event).filter(models.Event.event_id == event_id).first()
        if row is None:
            raise HTTPException(404, f"Event {event_id} not found")
        return _replay_for(row)
    finally:
        db.close()


def _replay_for(row: models.Event) -> dict:
    """Build a replay payload: actual trajectory + correct-technique path."""
    from pipeline.replay.generator import build_replay, replay_to_dict
    from pipeline.types import BehaviorMatch, PhysicsState, RiskEvent, Trajectory, TrajectoryPoint
    from pipeline.matching.dtw_matcher import ExemplarLibrary

    # Rehydrate the event for the replay builder
    ev = RiskEvent(
        event_id=row.event_id,
        timestamp_sec=row.timestamp_sec,
        frame_idx=row.frame_idx,
        source_video=row.source_video,
        behavior_class=row.behavior_class,
        risk_score=row.risk_score,
        risk_level=row.risk_level,
        dtw_match=BehaviorMatch(row.behavior_class, row.dtw_distance or 1.0,
                                row.confidence or 0.5, "", ""),
        physics=PhysicsState(0, physics_valid=row.physics_valid,
                             severity_score=row.physics_severity or 0.0),
    )
    # Actual trajectory from stored points (if any)
    traj = None
    if row.trajectories:
        t = row.trajectories[0]
        pts = [
            TrajectoryPoint(
                frame_idx=int(p.get("frame_idx", 0)),
                timestamp_sec=float(p.get("timestamp_sec", 0.0)),
                x=float(p.get("x", 0.0)),
                y=float(p.get("y", 0.0)),
                z_est=float(p.get("z_est", 0.0)),
            )
            for p in (t.points or [])
        ]
        traj = Trajectory(track_id=t.track_id, points=pts, class_label=t.class_label)
    # Correct-technique path from the matched exemplar
    correct = None
    try:
        lib = ExemplarLibrary()
        for e in lib.by_class(row.behavior_class):
            correct = e.get("points", [])
            break
    except Exception:
        correct = None
    replay = build_replay(ev, trajectory=traj, correct_points=correct)
    return replay_to_dict(replay)


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------

@app.post("/api/voice/generate", response_model=VoiceResponse)
async def voice_generate(req: VoiceRequest):
    """Generate a coaching message for a behavior/level (TTS behind flag)."""
    from pipeline.voice.coach import VoiceCoach

    vc = VoiceCoach(enabled=False)
    msg = vc.build_message(
        _pseudo_event(req.behavior_class, req.risk_level, req.physics_gated)
    )
    return VoiceResponse(message=msg, spoken=req.speak and False)  # never auto-speak


def _pseudo_event(behavior_class: str, risk_level: str, physics_gated: bool):
    from pipeline.types import RiskEvent

    ev = RiskEvent(event_id="", timestamp_sec=0.0, frame_idx=0, source_video="",
                   behavior_class=behavior_class, risk_score=0.0, risk_level=risk_level)
    ev.metadata = {"gated_by_physics": physics_gated}
    return ev


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------

@app.post("/api/assistant", response_model=AssistantResponse)
async def assistant_chat(req: AssistantRequest):
    """Chat with the RAG assistant (structured event queries only)."""
    from pipeline.assistant.rag import Assistant

    asm = Assistant()
    answer = asm.chat(req.message)
    return AssistantResponse(message=answer)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    # Feedback FK references events.id (UUID); the API exposes the human-facing
    # event_id, so resolve it to the row's DB id first.
    fk = None
    if req.event_id:
        ev = db.query(models.Event).filter(models.Event.event_id == req.event_id).first()
        if ev is None:
            raise HTTPException(404, f"Event {req.event_id} not found")
        fk = ev.id
    row = models.Feedback(
        event_id=fk,
        annotator=req.annotator,
        correct_behavior=req.correct_behavior,
        correct_risk_level=req.correct_risk_level,
        notes=req.notes,
    )
    db.add(row)
    db.commit()
    return {"ok": True, "feedback_id": str(row.id)}


# ---------------------------------------------------------------------------
# Health / metadata
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "app": "ReplayTwin"}


@app.get("/api/behaviors")
async def behaviors() -> dict:
    from pipeline.types import BEHAVIOR_CLASSES

    return {"behavior_classes": BEHAVIOR_CLASSES}