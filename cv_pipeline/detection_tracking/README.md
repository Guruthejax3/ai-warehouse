# Detection & Tracking Module

Stage 2 of the cv-pipeline. Maintains persistent object identities across
frames (`MOG2` foreground detection + `IoU` association + per-track **Kalman**
smoothing), so the kinematics stage sees stable trajectories, not per-frame
noise.

## Files

- `detection_tracking.py` — `Detector` (foreground + person/object label),
  `Tracker` (predict → associate → spawn/retire), `iou()`, `tracks_to_json()`
- `run_detection_tracking.py` — CLI entrypoint

## Run

```bash
python -m cv_pipeline.detection_tracking.run_detection_tracking \
    --video clip.mp4 --output tracks.json --fps 10
```

## Output schema (one entry per frame)

```json
{
  "frame_idx": int,
  "timestamp_sec": float,
  "tracks": [
    {
      "track_id": int,
      "class_label": "person" | "object",
      "bbox": [x1, y1, x2, y2],
      "center": [cx, cy],
      "area": float,
      "dropped": bool
    }
  ]
}
```

## Design notes

- **Re-ID**: a Kalman filter per track predicts the next state; detections are
  greedily associated by IoU (`iou_threshold=0.15`). Opaque tracks retire after
  `max_misses` unframed frames and carry a `dropped` flag — the moment an
  operator "drops" tracking.
- **Person / object split**: OpenCV HOG people detector when available,
  aspect-ratio heuristic otherwise — no extra installs.