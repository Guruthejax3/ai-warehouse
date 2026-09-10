# Risk Scoring Module

Stage 6 (final) of the cv-pipeline. Combines all prior stage evidence into a
single, auditable, real-time risk score per event:

```
risk = w_physics * kinematics_severity
     + w_pose    * pose_hazard
     + w_sequence * fsm_unsafe_sequence
     + w_fragility * fragility_exposure
```

Default weights: physics 0.45, pose 0.15, sequence 0.25, fragility 0.15.
`risk_score` (0–1) maps to `low` / `medium` / `high` / `critical`.

Every event carries its weight breakdown and a plain-English justification that
names the contributing evidence — fully traceable, no black box.

## Run

```bash
python -m cv_pipeline.risk_scoring.run_risk_scoring \
    --kinematics kinematics.json --fsm fsm.json --output events.json
```

## Output schema (per event)

```json
{
  "event_id": str,
  "track_id": int,
  "timestamp_sec": float,
  "behavior_class": "unsafe_loading_sequence" | "product_dropped" | "...",
  "risk_score": float,
  "risk_level": "low" | "medium" | "high" | "critical",
  "weights": {"physics": float, "pose": float, "sequence": float, "fragility": float},
  "justification": "kinematics severity 0.62, unsafe loading sequence (FSM violation), fragile product exposure."
}
```