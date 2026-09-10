# ReplayTwin — Feature Audit vs. Hackathon Expected Solution

**Audit date:** 2026-09-10
**Branch:** `gurutheja`
**Codebase:** Complete read of all pipeline, backend, frontend, config, schema, and exemplar files.

## Legend

| Mark | Meaning |
|------|---------|
| ✅ | Fully implemented, meets the requirement |
| ⚠️ | Partially implemented — works but has gaps or is not fully wired |
| ❌ | Missing or stubbed — no real implementation exists |

---

## Feature 1 — AI Video Understanding

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1.1 | **Ingest live or recorded video** | ✅ | `pipeline/runner.py:process_clip()` reads any mp4 via OpenCV. `backend/main.py:POST /api/ingest` receives uploads, calls `process_clip()`. Configurable source dir in `config.yaml:video_source`. |
| 1.2 | **Detect people, products, pallets and equipment** | ⚠️ | `pipeline/perception/segmentation.py:_classify_region()` detects people via face cascade. All non-person objects are labeled `"object"` — no pallet vs. product vs. equipment distinction. SAM2 (optional) is class-agnostic. |
| 1.3 | **Identify loading/unloading activities and idle time** | ❌ | No activity classifier exists. `pipeline/runner.py` processes trajectories but does not infer loading vs. unloading vs. idle from trajectory direction (e.g., toward vs. away from truck). No `activity_type` field on events. |
| 1.4 | **Track objects across frames** | ✅ | `pipeline/trajectory/builder.py:TrajectoryBuilder` accumulates IoU-associated detections across frames with Kalman smoothing. Handles occlusion gaps up to `max_gap_frames` (default 5). Sealed tracks become full trajectories. |

**Feature 1 summary:** 2 ✅, 1 ⚠️, 1 ❌ — **needs implementation**

---

## Feature 2 — Behaviour Detection

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 2.1 | **Detect dropping, dragging, throwing, improper stacking, unsafe loading** | ✅ | `pipeline/matching/dtw_matcher.py:DTWMatcher.match_trajectory()` detects `product_dropped`, `product_dragged`, `material_pushed_or_thrown` via DTW against exemplar library. `pipeline/rules/zones.py` fires `outside_designated_zone`. `physics/kinematics.py:cross_check()` validates `rough_handling` (jerk) and `unsafe_loading_sequence` (height change). 10 behavior classes defined in `pipeline/types.py:BEHAVIOR_CLASSES`. |
| 2.2 | **Distinguish normal from potentially damaging behaviour** | ✅ | `pipeline/risk/scoring.py:RiskScorer.score_event()` computes a weighted risk score (DTW match + physics severity + fragility + repetition + zone criticality). Physics cross-check (`mandatory_before_alert`) gates alerts — behaviors without confirming kinematics are capped at "low". Each event carries a detailed human-readable justification string. |
| 2.3 | **Detect sequence of actions rather than individual frames** | ⚠️ | DTW matching operates on full trajectory windows (not single frames) — `extract_trajectory_vector()` resamples to 32 points across the trajectory's lifetime. `min_track_length: 10` (~2s at 5fps) enforces multi-frame evidence. However: (a) no explicit *multi-step sequence* detector (e.g., "approach → lift → load" as a chained FSM), and (b) zone rule checks only the final centroid position, not temporal progression. |

**Feature 2 summary:** 2 ✅, 1 ⚠️ — **needs enhancement for sequence-of-actions detection**

---

## Feature 3 — Damage-Risk Detection

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 3.1 | **Assign risk level to each event** | ✅ | `pipeline/risk/scoring.py:_classify()` maps 0–1 score to `low` / `medium` / `high` / `critical` thresholds (0.25 / 0.50 / 0.75). Every `RiskEvent` carries `risk_score` and `risk_level`. |
| 3.2 | **Identify high-risk events requiring immediate attention** | ✅ | Events with `risk_level in ["high", "critical"]` are surfaced first — `runner.py:182` sorts events by `risk_score` descending. `backend/main.py:GET /api/events` returns all events; frontend `LiveEventFeed.tsx` displays risk badges. |
| 3.3 | **Highlight potential damage-causing actions** | ✅ | Each event has a 10-class `behavior_class` identifying the specific damaging action (`product_dropped`, `rough_handling`, etc.). `RiskEvent.justification` explains *why* it's risky in plain English. |
| 3.4 | **Generate event timestamp** | ✅ | `RiskEvent.timestamp_sec` is set from the source frame's synthetic clock. DB `events` table has `timestamp_sec` column. |
| 3.5 | **Generate evidence clip with timestamps** | ⚠️ | `pipeline/risk/scoring.py:_evidence_window()` computes `evidence_clip_start` / `evidence_clip_end` frame numbers (±5s). `pipeline/privacy/blur.py:extract_evidence_clip()` trims and face-blurs the clip, writes mp4 to `data/evidence_clips/`. `backend/main.py:ingest()` calls `extract_evidence_clip()` and sets `evidence_clip_path` on the DB row. **Gap:** `evidence_clip_path` is NOT set on the `RiskEvent` dataclass *before* DB persist — the backend sets it only on the SQLAlchemy model after extraction, so in-memory events returned from `process_clip()` lack the path. Also, `RiskEvent` dataclass has `evidence_clip_path` as an empty string default — it's never populated by the pipeline itself. |

**Feature 3 summary:** 4 ✅, 1 ⚠️ — **evidence clip path needs wiring into RiskEvent**

---

## Feature 4 — AI Operations Assistant

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 4.1 | **Explain what happened in simple terms** | ✅ | `pipeline/assistant/rag.py:EventStore.explain_event()` returns a full event record. The LLM (via Anthropic Tool Runner) receives this data and can explain it in natural language. `backend/main.py:POST /api/assistant` sends the query + tool results to Claude. |
| 4.2 | **Explain why certain behaviour may be risky** | ✅ | The `explain_event` tool returns the event's `justification` field (human-readable explanation from `risk/scoring.py:_build_justification()`), physics measurements, and risk breakdown. The LLM synthesizes this into a natural-language risk explanation. |
| 4.3 | **Answer questions about incidents** | ✅ | `query_high_risk_events` tool filters by risk level + time range. `query_behavior_stats` returns per-class counts. `query_bay_risk` aggregates by bay/zone. The LLM can answer "what happened at bay 3?" or "how many drops today?" via these tools. |
| 4.4 | **Recommend corrective actions** | ❌ | No corrective action recommendation tool or knowledge base exists. The LLM could *improvise* recommendations from its general knowledge, but the requirement implies structured, system-specific recommendations (e.g., "operator X should repeat pallet-stacking training"). No `corrective_actions` table or lookup module. |
| 4.5 | **Summarize shift-level observations** | ❌ | No `summarize_shift` tool. `EventStore` has no shift-aware queries — no `shift_id` column in the schema, no shift-based aggregation, no end-of-shift summary endpoint. |

**Feature 4 summary:** 3 ✅, 2 ❌ — **needs corrective actions tool + shift summary tool**

---

## Feature 5 — Visual Alerts & Dashboard

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 5.1 | **Highlight detected event on video** | ⚠️ | `frontend/components/ReplayViewer.tsx` renders a canvas animation showing the actual trajectory (red) vs. correct technique (green) with time slider. This is a trajectory replay, *not* video playback with an overlay. No video-with-bounding-box playback exists. |
| 5.2 | **Display risk category visually** | ✅ | `LiveEventFeed.tsx` shows color-coded risk badges (`bg-red-500` for critical, etc.). `BayHeatmap.tsx` colors grid cells by worst risk level. `SafetyScorecard.tsx` shows high/critical counts. |
| 5.3 | **Show event timeline** | ⚠️ | `LiveEventFeed.tsx` renders a list of events sorted by time, but it's a flat feed — not a visual *timeline* with time axis, zoom, or drill-down. No time-series visualization (e.g., events over hours with a scrubber). |
| 5.4 | **Generate daily/shift summary report** | ❌ | No summary report component or endpoint. `SafetyScorecard.tsx` shows aggregate KPIs (total events, avg risk, top behavior) but there's no daily/shift breakdown, no report generation, no date filtering. |
| 5.5 | **Provide behaviour trends** | ✅ | `BehaviorTrends.tsx` renders a Recharts `BarChart` showing event counts per behavior class. |

**Feature 5 summary:** 2 ✅, 2 ⚠️, 1 ❌ — **needs timeline view + shift summary report**

---

## Feature 6 — Prevention & Learning

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 6.1 | **Identify recurring behaviours** | ❌ | No aggregation endpoint for recurring behavior patterns. `EventStore.query_behavior_stats()` returns raw per-class counts but: (a) no "recurring" detection (e.g., same behavior at same bay multiple times), (b) no API endpoint exposes this, (c) frontend has no recurring-behavior view. `backend/main.py` lacks any analytics endpoint. |
| 6.2 | **Recommend targeted training opportunities** | ❌ | No training recommendation system. No mapping of behavior classes → training modules, no operator-level tracking, no recommendation endpoint or UI component. |
| 6.3 | **Highlight high-risk locations/processes** | ✅ | `BayHeatmap.tsx` shows a grid of 8 bays colored by worst risk level. `query_bay_risk` tool in the assistant provides bay-level risk aggregation. |
| 6.4 | **Track improvement over time** | ❌ | No trend-tracking. `BehaviorTrends.tsx` shows counts per class (no time dimension). No date-range filtering, no week-over-week comparison, no improvement metric, no trend endpoint. DB schema has no `shift_id` or date-bucketing support. |

**Feature 6 summary:** 1 ✅, 3 ❌ — **needs recurring behavior API, training recommendations, trend tracking**

---

## Overall Summary

| Feature | ✅ | ⚠️ | ❌ | Action needed |
|---------|----|----|----|--------------|
| 1 — AI Video Understanding | 2 | 1 | 1 | Activity classifier (loading/unloading/idle), object sub-classification |
| 2 — Behaviour Detection | 2 | 1 | 0 | Multi-step sequence detector (FSM for loading sequences) |
| 3 — Damage-Risk Detection | 4 | 1 | 0 | Wire evidence_clip_path into RiskEvent before DB persist |
| 4 — AI Operations Assistant | 3 | 0 | 2 | `corrective_actions` tool + knowledge base, `summarize_shift` tool |
| 5 — Visual Alerts & Dashboard | 2 | 2 | 1 | Video overlay replay, timeline view, shift summary report |
| 6 — Prevention & Learning | 1 | 0 | 3 | Recurring behavior endpoint, training recommendations, trend tracking |
| **Total** | **14** | **5** | **7** | **26 items** |

---

## Implementation Priority Order

1. **Feature 3** (1 ⚠️) — Wire `evidence_clip_path` into `RiskEvent`. Quick fix.
2. **Feature 2** (1 ⚠️) — Add loading-sequence FSM detector.
3. **Feature 1** (1 ⚠️ + 1 ❌) — Activity classifier + object sub-classification.
4. **Feature 4** (2 ❌) — `corrective_actions.py` + `summarize_shift` tool.
5. **Feature 5** (2 ⚠️ + 1 ❌) — Timeline view, shift summary component, video overlay.
6. **Feature 6** (3 ❌) — Recurring behavior API, training recs, trend tracking.
