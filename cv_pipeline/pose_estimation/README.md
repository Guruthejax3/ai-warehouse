# Pose Estimation Module

Stage 3 of the cv-pipeline. Estimates human body keypoints and posture flags
to identify hazardous worker positions (bending, reaching, stooping).

Two backends, both CPU-friendly:

- **`box`** (default, zero dependencies): derives a 14-keypoint skeleton from
  the person bbox proportions + optical-flow velocity, then computes the same
  hazard flags from geometry.
- **`mediapipe`**: MediaPipe Pose (33 landmarks). Auto-falls back to `box` if
  the wheel is missing or fails on this host.

## Run

```bash
python -m cv_pipeline.pose_estimation.run_pose_estimation \
    --video clip.mp4 --tracks tracks.json --output poses.json \
    --backend box
```

## Output schema

```json
{
  "track_id": int,
  "frame_idx": int,
  "timestamp_sec": float,
  "posture": {
    "hazardous": bool,
    "flags": {"bending": bool, "reaching": bool, "stooping": bool},
    "elbow_angle": float | null,
    "justification": "Pose: hazardous — trunk bending (extends torso)."
  }
}
```

These flags feed the risk-scoring stage (weight `w_pose`).