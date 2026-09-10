# ReplayTwin — Warehouse Video-Intelligence

**AI field-intelligence for warehouse handling.** ReplayTwin watches warehouse
footage, detects unsafe handling of products in real time, scores each event's
risk with an *explainable justification*, replays the actual vs. correct
technique, and answers natural-language questions — **only** from the structured
event log, never by free-generating facts.

Built for a hackathon on a single laptop: optional heavy models (SAM2, MediaPipe,
GPU) degrade gracefully to OpenCV fallbacks, and Postgres degrades to SQLite.

---

## Architecture

```
Video (.mp4)
   │  OpenCV VideoCapture, frame-skip to target FPS
   ▼
Motion detection ──► flow-flagged bboxes            cv_pipeline/motion_detection/
   │                                                    (Farneback / RAFT — NOT rewritten)
   ▼
Segmentation + tracking ──► TrackedObject(id, bbox)  pipeline/perception/segmentation.py
   │                       (SAM2 on flow bboxes, or OpenCV IoU+Kalman)
   ▼
Pose (optional) ──► keypoints                          pipeline/perception/pose.py
   ▼
Trajectory builder ──► {(t,x,y,z_est) per track}      pipeline/trajectory/builder.py
   ▼
Behaviour matching ──► DTW vs exemplar library (scoped      pipeline/matching/dtw_matcher.py
   │                     to motion-shape classes: drop/drag/throw)
   ├── Rule-based behaviours ──► zone placement, layout       pipeline/rules/zones.py
   │                            (deterministic rules, NOT DTW)
   ▼
Physics check ──► v, a, jerk, drop height              pipeline/physics/kinematics.py
   │             (mandatory gate before alerting; throw/push uses launch
   │              signature — horizontal throws pass without a drop height)
   ▼
Risk scoring ──► score + risk level + justification    pipeline/risk/scoring.py
   │            persist to Postgres; trim ±5s evidence clip, blur faces
   ▼
Replay ──► actual + correct trajectories                pipeline/replay/generator.py
Voice ──► optional TTS coaching (behind flag)           pipeline/voice/coach.py
Assistant ──► RAG over Postgres event log only          pipeline/assistant/rag.py
   ▼
FastAPI backend + WebSocket                             backend/main.py
   ▼
Next.js dashboard + 2D replay viewer                    frontend/
```

### Risk scoring — explainable by design

```
risk_score = w1·(1 − dtw_distance_normalized)   behaviour match quality
           + w2·physics_severity                velocity / jerk / drop height
           + w3·fragility_class                 standard / fragile / very fragile
           + w4·repetition_count                how often the track repeats
           + w5·zone_criticality                how critical the bay is

Levels: low (<0.25) · medium (<0.50) · high (<0.75) · critical (≥0.75)
```

Every event **must** carry a human-readable justification (e.g.
`"Product dropped: downward velocity 2.4 m/s exceeded threshold 2.0 m/s; DTW
match to product_dropped at 0.87; fragile class 0.7"`) — never a bare confidence
number. Physics is a mandatory gate: if the kinematics cross-check fails, the
alert is blocked (score capped, flagged `gated_by_physics`).

### Privacy

Only a **±5 s evidence clip** around each event is retained, with **faces
blurred** (Haar cascade + full-frame fallback). Full raw video is never stored
or served by the API.

---

## Repository layout

```
backend/            FastAPI app (ingest, events, replay, voice, assistant, WS)
cv_pipeline/        six standalone CV stages (motion → tracking → pose → kinematics → FSM → risk)
data/clips/         real pilot videos (GITIGNORED — never committed)
data/exemplars/     behaviour exemplar library (JSON); 2 real pilot-clip exemplars
data/evidence_clips/±5s blurred evidence clips (GITIGNORED)
db/schema.sql       Postgres schema (events, trajectories, feedback, …)
frontend/           Next.js dashboard + replay viewer
pipeline/           the downstream pipeline (perception → … → risk → assistant)
pipeline/rules/     deterministic rule-based behaviours (zones; stacking stubs)
scripts/            CLI helpers (run_pipeline.py, tag_exemplar.py)
tests/              unit + e2e tests
config.yaml         pipeline tuning (FPS, thresholds, weights, exemplars, zones)
```

---

## Setup

### 1. Python pipeline + backend (Python 3.10+; developed on 3.14/Windows)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

Optional (graceful fallbacks if absent): `torch` + SAM2, `mediapipe`,
`pyttsx3`.

### 2. Copy real clips

```bash
mkdir -p data/clips
# copy your pilot .mp4 files into data/clips/  (gitignored, never committed)
```

### 3. Run the pipeline on a clip

```bash
python scripts/run_pipeline.py "data/clips/Throwing Mattresses.mp4"
```

### 4. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

- `DATABASE_URL` (Postgres) is the intended store; if unset/unreachable the
  backend falls back to SQLite (set `REPLAYTWIN_DB_SQLITE=data/backend.db`).
- Ingest a clip: `POST /api/ingest` — events stream live over `WS /ws/events`.

### 5. Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000  (API defaults to :8000)
```

### Docker (full stack)

```bash
cp .env.example .env   # fill ANTHROPIC_API_KEY if you want the assistant
docker compose up --build
# frontend :3000 · backend :8000 · postgres :5432
```

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/ingest` | Run pipeline on a video, return events |
| GET | `/api/events` | Query events (filters: risk, behaviour, bay, time) |
| GET | `/api/events/{id}` | Single event detail (+ replay data) |
| GET | `/api/replay/{id}` | Actual + correct trajectories |
| GET | `/api/behaviors` | Behaviour counts |
| POST | `/api/voice/generate` | TTS coaching message (behind flag) |
| POST | `/api/assistant` | Chat with the RAG assistant |
| POST | `/api/feedback` | Submit feedback on an event |
| GET | `/api/health` | Health check |
| WS | `/ws/events` | Live event stream |

The assistant's tools (`query_high_risk_events`, `query_behavior_stats`,
`query_bay_risk`, `explain_event`) query **only** the structured event log.

---

## Testing

```bash
python -m pytest -q
```

- `test_dtw_matcher.py` — DTW matching on synthetic trajectories
- `test_risk_scoring.py` — risk formula, levels, justification, physics gate
- `test_kinematics.py` — 2D drop-height estimation, throw vs. drag signatures
- `test_zones.py` — deterministic zone-placement rule (normalised polygons)
- `test_replay_assistant_voice.py` — replay generator, assistant tools, voice
- `test_backend.py` — FastAPI endpoints, WS broadcast, feedback (SQLite)
- `test_pipeline_e2e.py` — real clip through the full pipeline (skips if
  `data/clips/` is empty)

---

## Known limitations / TODOs

- **SAM2 & MediaPipe are optional** — no wheels on Python 3.14 Windows, so the
  OpenCV IoU+Kalman tracker and skip-pose fallback are what actually run here.
- **No GPU** — all models run on CPU. SAM2 (if installed) is ~0.5–2 s/frame on
  CPU, so it is limited to flow-flagged regions and low target FPS.
- **Voice coaching is a stub** behind `VOICE_COACHING_ENABLED`.
- **Rule-based behaviours are partially implemented.** `outside_designated_zone`
  is fully wired (per-camera normalised polygons; default zone covers the frame
  so the rule is dormant until operators configure real bays). `incorrect_stacking`,
  `unstable_stacking`, `no_required_equipment`, `pallet_mispositioned`, and
  `unsafe_loading_sequence` are acknowledged signatures in the exemplar library
  but lack dedicated rule/state detectors — exercises are placeholders until a
  stack-cluster / equipment-proximity / sequence FSM exists. They are excluded
  from the DTW competition (`matching.dtw_classes`) so they cannot swallow real
  shape matches.
- **Exemplar library** ships 2 real tracks tagged from pilot clips
  (`product_dragged`, `material_pushed_or_thrown`); the rest are synthetic seeds.
  Use `scripts/tag_exemplar.py` to tag more from your footage.
- **Homography / pixel-scale is manual** — calibrate reference-object dimensions
  in `config.yaml` per camera.
- **Trajectory-only classification** can call a long, straight carry a "drag"
  (no person/object discrimination upstream). The physics gate + risk threshold
  filter most noise; a per-camcorder calibration pass and zone config are
  expected before production use.
- **Only 7 pilot clips** are available for testing.
- **No CI/CD** yet.
