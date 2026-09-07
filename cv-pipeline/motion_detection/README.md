# Motion Detection Module

Detects regions of significant motion per frame using dense optical flow.

## Purpose

First stage of the Damage-DNA CV pipeline, upstream of object detection/tracking.
Flags moving regions per frame and classifies each event as:

- **sustained** — low-magnitude long-duration motion (dragging, rolling)
- **sudden_spike** — high-magnitude short-duration motion (throwing, dropping)
- **none** — no significant motion

## Running

Single video:
```bash
python -m cv-pipeline.motion_detection.run_motion_detection \
    --video "data/pilot_videos/Dock level, dragging cupboard.mp4" \
    --output motion_output.json \
    --fps 10 \
    --visualize
```

Batch (all videos in data/pilot_videos/):
```bash
python -m cv-pipeline.motion_detection.run_motion_detection \
    --batch \
    --fps 10
```

## Output Schema
```json
{
    "frame_idx": int,
    "timestamp_sec": float,
    "motion_regions": [{"bbox": [x1,y1,x2,y2], "magnitude": float}],
    "frame_motion_score": float,
    "event_type": "sustained" | "sudden_spike" | "none"
}
```

## How event_type is determined

A rolling mean + std of frame_motion_score is maintained over a configurable
window (default 15 frames). If the current score exceeds rolling_mean +
spike_std_multiplier * rolling_std, the event is tagged `sudden_spike`.
Otherwise, if above motion_threshold it is `sustained`.

## Switching backends

- Farneback (default, no extra deps): `--method farneback`
- RAFT (requires torch+torchvision): `--method raft`

## CLI flags
- `--video`, `--output`, `--method`, `--fps`, `--threshold`, `--min-area`,
  `--spike-std-multiplier`, `--rolling-window`, `--visualize`, `--batch`,
  `--output-dir`