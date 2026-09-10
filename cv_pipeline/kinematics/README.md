# Kinematics Module

Stage 4 of the cv-pipeline. Computes physical motion parameters from tracked
object trajectories: **velocity, acceleration, jerk, drop-height estimate,
vertical thrust, stop latency, and collision warnings** — all converted to SI
units through `pixel_scale` (metres per pixel).

## Run

```bash
python -m cv_pipeline.kinematics.run_kinematics \
    --tracks tracks.json --output kinematics.json --pixel-scale 0.01
```

## Output schema (per track)

```json
{
  "track_id": int,
  "sample_count": int,
  "peak_speed_mps": float,
  "mean_speed_mps": float,
  "peak_accel_mps2": float,
  "peak_jerk_mps3": float,
  "drop_height_est_m": float,
  "vertical_thrust_mps": float,
  "stop_latency_ms": float,
  "collision_warning": bool,
  "severity_score": float,
  "justification": "Kinematics: peak speed 1.3 m/s, estimated drop height 0.6 m."
}
```

`severity_score` (0–1) is a weighted physics snapshot — the primary input to
the risk-scoring stage (`w_physics`).