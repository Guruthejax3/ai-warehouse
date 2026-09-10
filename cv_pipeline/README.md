# cv-pipeline — six runnable CV stages

The stage-by-stage computer-vision implementation behind **ReplayTwin**.
Every stage is a standalone, dependency-light module with its own CLI and a
JSON handoff contract, so a reviewer can run each one individually — or chain
all six with the orchestrator below.

```
┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ 1 motion     │ → │ 2 detect+track│ → │ 3 pose       │
│    regions   │   │  (Kalman IoU) │   │  + posture   │
└──────────────┘   └───────────────┘   └──────────────┘
        │                │                    │
        ▼                ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ 4 kinematics │ → │ 5 behaviour  │ → │ 6 risk scoring   │
│  physics     │   │  FSM / seq   │   │  explainable 0-1 │
└──────────────┘   └──────────────┘   └──────────────────┘
```

| # | Stage | Module | What it produces |
|---|-------|--------|------------------|
| 1 | Motion detection | `motion_detection/` | motion regions + `sustained` / `sudden_spike` event type |
| 2 | Detection & tracking | `detection_tracking/` | persistent object tracks (MOG2 + IoU + Kalman) |
| 3 | Pose estimation | `pose_estimation/` | 14-keypoint skeleton + hazard-posture flags (MediaPipe or zero-dep "box") |
| 4 | Kinematics | `kinematics/` | velocity / acceleration / jerk / drop-height / collision warning |
| 5 | Behaviour FSM | `behaviour_fsm/` | activity state machine + unsafe loading-sequence violations |
| 6 | Risk scoring | `risk_scoring/` | explainable weighted risk score + risk_level per event |

## Run everything end-to-end (no clip needed)

```bash
# Synthetic 90-frame clip — demos the whole chain without a video file
python -m cv_pipeline.run_pipeline --smoke --output reports/cv_pipeline_report.json

# Real clip
python -m cv_pipeline.run_pipeline --video data/clips/"a clip.mp4" --output reports/cv_pipeline_report.json

# See the stage diagram only
python -m cv_pipeline.run_pipeline --demo
```

## Run a single stage

```bash
# Stage 1
python -m cv_pipeline.motion_detection.run_motion_detection --video clip.mp4 --output motion.json

# Stage 2 (consumes raw video)
python -m cv_pipeline.detection_tracking.run_detection_tracking --video clip.mp4 --output tracks.json

# Stage 3 (video + stage-2 JSON)
python -m cv_pipeline.pose_estimation.run_pose_estimation --video clip.mp4 --tracks tracks.json --output poses.json

# Stage 4 (stage-2 JSON)
python -m cv_pipeline.kinematics.run_kinematics --tracks tracks.json --output kinematics.json

# Stage 5 (stage-4 JSON)
python -m cv_pipeline.behaviour_fsm.run_behaviour_fsm --kinematics kinematics.json --output fsm.json

# Stage 6 (stage-4 + stage-5 JSON)
python -m cv_pipeline.risk_scoring.run_risk_scoring --kinematics kinematics.json --fsm fsm.json --output events.json
```

## How it relates to `pipeline/`

`pipeline/` is the integrated orchestrator the backend (`POST /api/ingest`)
uses: it wires these primitives plus the exemplar/DTW matcher, physics gate,
activity classifier, interaction monitor, and RAG assistant into one pass.
`cv-pipeline/` is the reviewer-facing, stage-by-stage implementation — the
same science, each stage runnable and inspectable on its own.

Run any stage with `PYTHONPATH=.` from the repo root.