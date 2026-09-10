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
from datetime import datetime, timedelta, timezone
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
    """Generate a coaching message for a behavior/level (TTS behind flag).

    ``language`` selects the alert locale: en | hi | te | es (multilingual
    voice alerts innovation feature). Spoken audio stays behind a flag.
    """
    from pipeline.voice.coach import VoiceCoach

    vc = VoiceCoach(enabled=False, language=req.language)
    msg = vc.build_message(
        _pseudo_event(req.behavior_class, req.risk_level, req.physics_gated),
        language=req.language,
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


@app.get("/api/evidence/{event_id}")
async def serve_evidence_clip(event_id: str):
    """Serve the face-blurred, trimmed evidence clip for an event."""
    from fastapi.responses import FileResponse

    db = next(get_db())
    try:
        row = db.query(models.Event).filter(models.Event.event_id == event_id).first()
        if row is None or not row.evidence_clip_path:
            raise HTTPException(404, "Evidence clip not found")
        clip = Path(row.evidence_clip_path)
        if not clip.exists():
            raise HTTPException(404, "Evidence clip file not found on disk")
        return FileResponse(str(clip), media_type="video/mp4")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Analytics endpoints — shift summary, trends, corrective actions, zone risk
# ---------------------------------------------------------------------------

@app.get("/api/analytics/shift-summary")
async def shift_summary(window_days: int = 1):
    """Shift briefing: totals, per-behavior breakdown, worst bay, narrative."""
    from pipeline.assistant.rag import EventStore

    store = EventStore()
    return store.summarize_shift(window_days)


@app.get("/api/analytics/trend")
async def analytics_trend(window_days: int = 7):
    """Daily risk aggregate — improvement-over-time line chart data."""
    from pipeline.assistant.rag import EventStore

    store = EventStore()
    return {"days": store.trend(window_days), "window_days": window_days}


@app.get("/api/analytics/recurring")
async def analytics_recurring(window_days: int = 7, min_occurrences: int = 2):
    """Behaviors that recur across bays — cross-bay pattern detection."""
    from pipeline.assistant.rag import EventStore

    store = EventStore()
    return {"behaviors": store.recurring_behaviors(window_days, min_occurrences)}


@app.get("/api/analytics/corrective-actions")
async def list_all_corrective_actions():
    """Full corrective action knowledge base (all 10 behavior classes)."""
    from pipeline.assistant.corrective_actions import all_recommendations

    return {"recommendations": all_recommendations()}


class CorrectiveActionRequest(BaseModel):
    behavior_class: str
    risk_level: str = "medium"


@app.post("/api/analytics/corrective-action")
async def get_corrective_action(req: CorrectiveActionRequest):
    """Corrective action for a specific behavior + risk level."""
    from pipeline.assistant.corrective_actions import recommend_for

    rec = recommend_for(req.behavior_class, req.risk_level)
    if rec is None:
        raise HTTPException(404, f"No corrective action for '{req.behavior_class}'")
    return rec


@app.get("/api/analytics/zone-risk")
async def zone_risk_aggregation(window_days: int = 7):
    """Risk-by-zone aggregation: sum, max, avg per bay for the heatmap."""
    from pipeline.assistant.rag import EventStore

    store = EventStore()
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=max(0, window_days))).strftime('%Y-%m-%d %H:%M:%S.%f')
    sql = (
        "SELECT zone_id, COUNT(*) AS events, MAX(risk_score) AS max_risk, "
        "AVG(risk_score) AS avg_risk, "
        "SUM(CASE WHEN risk_level IN ('high','critical') THEN 1 ELSE 0 END) "
        "AS severe_count "
        "FROM events WHERE created_at >= ? "
        "GROUP BY zone_id ORDER BY max_risk DESC"
    )
    with store.engine.connect() as conn:
        rows = conn.exec_driver_sql(sql, (cutoff,)).fetchall()
        cols = list(rows[0]._mapping.keys()) if rows else []
        zones = [dict(zip(cols, r)) for r in rows]
    return {"zones": zones, "window_days": window_days}


@app.get("/api/analytics/training")
async def training_recommendations():
    """Training recommendations: per-behavior training units with category."""
    from pipeline.assistant.corrective_actions import CORRECTIVE_ACTIONS

    training = []
    for cls, action in CORRECTIVE_ACTIONS.items():
        training.append({
            "behavior_class": cls,
            "training_unit": action["training_unit"],
            "category": action["category"],
            "process_fix": action["process_fix"],
        })
    return {"training_units": training}


# ---------------------------------------------------------------------------
# Innovation endpoints — damage prediction, incident reports, pallet stability
# ---------------------------------------------------------------------------

class DamagePredictionRequest(BaseModel):
    risk_score: float = 0.5
    behavior_class: str = "product_dropped"
    fragility: str = "standard"
    physics_severity: float = 0.0
    confidence: float = 0.5
    history_count: int = 0


@app.post("/api/analytics/damage-prediction")
async def damage_prediction(req: DamagePredictionRequest):
    """Score the probability of product damage from event context."""
    from pipeline.innovation.damage_prediction import predict_damage

    return predict_damage(
        risk_score=req.risk_score,
        behavior_class=req.behavior_class,
        fragility=req.fragility,
        physics_severity=req.physics_severity,
        confidence=req.confidence,
        history_count=req.history_count,
    ).__dict__


class IncidentReportRequest(BaseModel):
    event_id: str = ""
    behavior_class: str = "product_dropped"
    risk_level: str = "medium"
    risk_score: float = 0.5
    zone_id: str = "bay_0"
    source_video: str = ""
    timestamp_sec: float = 0.0
    justification: str = ""
    evidence_clip_path: str = ""
    activity_type: str = "unknown"
    confidence: Optional[float] = None


@app.post("/api/analytics/incident-report")
async def incident_report(req: IncidentReportRequest):
    """Generate a structured incident report for one event."""
    from pipeline.innovation.incident_report import (
        generate_incident_report, report_to_dict,
    )
    from pipeline.assistant.corrective_actions import recommend_for

    # Enrich with corrective action
    ca = recommend_for(req.behavior_class, req.risk_level)

    report = generate_incident_report(
        event_id=req.event_id,
        behavior_class=req.behavior_class,
        risk_level=req.risk_level,
        risk_score=req.risk_score,
        zone_id=req.zone_id,
        source_video=req.source_video,
        timestamp_sec=req.timestamp_sec,
        justification=req.justification,
        evidence_clip_path=req.evidence_clip_path,
        activity_type=req.activity_type,
        confidence=req.confidence,
        corrective_action=ca,
    )
    return report_to_dict(report)


class PalletStabilityRequest(BaseModel):
    trajectory_points: List[dict] = []
    pixel_scale: float = 0.01


@app.post("/api/pallet-stability")
async def pallet_stability(req: PalletStabilityRequest):
    """Assess pallet loading stability from trajectory physics."""
    from pipeline.innovation.pallet_stability import assess_pallet_stability

    if not req.trajectory_points:
        raise HTTPException(400, "trajectory_points required")
    result = assess_pallet_stability(req.trajectory_points, req.pixel_scale)
    return result.__dict__


class WMSAlertRequest(BaseModel):
    event_id: str
    behavior_class: str
    risk_level: str = "medium"
    risk_score: float = 0.5
    zone_id: str = "bay_0"
    justification: str = ""


@app.post("/api/wms/webhook")
async def wms_webhook(req: WMSAlertRequest):
    """Build and return a WMS safety alert (outbound webhook payload)."""
    from pipeline.innovation.wms_integration import build_wms_alert, alert_to_dict

    alert = build_wms_alert(
        event_id=req.event_id,
        behavior_class=req.behavior_class,
        risk_level=req.risk_level,
        risk_score=req.risk_score,
        zone_id=req.zone_id,
        justification=req.justification,
    )
    return alert_to_dict(alert)


class CCTVWebhookRequest(BaseModel):
    source_camera: str = "cam_01"
    event_type: str = "motion"
    zone_id: str = "bay_0"
    confidence: float = 0.5
    metadata: dict = {}


@app.post("/api/cctv/webhook")
async def cctv_webhook(req: CCTVWebhookRequest):
    """Parse an incoming CCTV/VMS event and return normalised data."""
    from pipeline.innovation.wms_integration import parse_cctv_webhook, cctv_to_dict

    event = parse_cctv_webhook(req.dict())
    return cctv_to_dict(event)


# ---------------------------------------------------------------------------
# Digital twin — lightweight warehouse state endpoint
# ---------------------------------------------------------------------------

@app.get("/api/digital-twin")
async def digital_twin_state(window_days: int = 1):
    """Warehouse digital twin: zone states, active hazards, object positions.

    Returns a lightweight state snapshot that the frontend renders as an
    isometric 3D warehouse view with heat overlays and object markers.
    """
    from pipeline.assistant.rag import EventStore

    store = EventStore()
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=max(0, window_days))).strftime('%Y-%m-%d %H:%M:%S.%f')

    # Zone states from recent events
    sql = (
        "SELECT zone_id, COUNT(*) AS events, MAX(risk_score) AS max_risk, "
        "AVG(risk_score) AS avg_risk "
        "FROM events WHERE created_at >= ? "
        "GROUP BY zone_id"
    )
    with store.engine.connect() as conn:
        rows = conn.exec_driver_sql(sql, (cutoff,)).fetchall()
        cols = list(rows[0]._mapping.keys()) if rows else []
        zone_data = [dict(zip(cols, r)) for r in rows]

    # Build zone states for all 8 bays
    BAYS = [f"bay_{i}" for i in range(8)]
    zones = {}
    for bay in BAYS:
        zd = next((z for z in zone_data if z["zone_id"] == bay), None)
        if zd:
            max_r = float(zd.get("max_risk", 0) or 0)
            zones[bay] = {
                "zone_id": bay,
                "events": int(zd.get("events", 0) or 0),
                "max_risk": round(max_r, 3),
                "avg_risk": round(float(zd.get("avg_risk", 0) or 0), 3),
                "status": ("critical" if max_r > 0.7 else
                           "warning" if max_r > 0.4 else "normal"),
            }
        else:
            zones[bay] = {
                "zone_id": bay, "events": 0, "max_risk": 0.0,
                "avg_risk": 0.0, "status": "normal",
            }

    # Active hazards: high/critical events in the last hour
    hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S.%f')
    hazard_sql = (
        "SELECT event_id, zone_id, behavior_class, risk_level, risk_score "
        "FROM events WHERE created_at >= ? AND risk_level IN ('high','critical') "
        "ORDER BY risk_score DESC LIMIT 10"
    )
    with store.engine.connect() as conn:
        hrows = conn.exec_driver_sql(hazard_sql, (hour_ago,)).fetchall()
        hcols = list(hrows[0]._mapping.keys()) if hrows else []
        hazards = [dict(zip(hcols, r)) for r in hrows]

    return {
        "zones": zones,
        "active_hazards": hazards,
        "total_events": sum(z["events"] for z in zones.values()),
        "overall_risk": round(
            max((z["max_risk"] for z in zones.values()), default=0.0), 3
        ),
        "window_days": window_days,
    }