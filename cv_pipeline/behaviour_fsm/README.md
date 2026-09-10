# Behaviour FSM Module

Stage 5 of the cv-pipeline. Models worker/equipment behaviour as a **finite
state machine** over activity labels (idle, dock_approach, handle, transit,
place, depart) and detects **unsafe state transitions** — the multi-step
*sequence* evidence a single-frame classifier cannot see.

## States & safe cycle

```
idle -> dock_approach -> handle -> transit -> place -> idle
```

Violations detected:
- `lift_without_approach` — handle with no prior approach (rushed lift)
- `depart_with_load` — leaving while still holding a load
- `rushed_placement` — placing at speed (impact risk)
- `illegal_<a>_to_<b>` — any other out-of-machine transition

## Run

```bash
python -m cv_pipeline.behaviour_fsm.run_behaviour_fsm \
    --kinematics kinematics.json --output fsm.json --min-dwell 0.4
```

## Output schema (per track)

```json
{
  "track_id": int,
  "states": [{"time": float, "state": str, "evidence": str}],
  "cycle_count": int,
  "violations": [{"time": float, "type": str, "evidence": str}],
  "unsafe_sequence": bool
}
```

`unsafe_sequence` feeds the risk-scoring stage (`w_sequence`).