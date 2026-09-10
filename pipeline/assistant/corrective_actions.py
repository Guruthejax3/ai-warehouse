"""Corrective actions — structured recommendations for each detected behavior.

The assistant's "recommend corrective actions" capability must be grounded in
a deterministic knowledge base, not improvised. Each behavior class maps to:

    - immediate_action   : what the operator should do right now
    - process_fix        : the process/technique change that prevents repeats
    - training_unit      : which training module targets this behavior
    - severity_gate      : the minimum risk level at which the action is auto-
                           recommended (loud alarms, supervisor escalation)

The knowledge base doubles as the *training recommendation* source (Feature 6):
the training_unit per class is returned by the ``recommend_training`` endpoint
and the assistant's ``list_corrective_actions`` tool, so targeted coaching is
always tied to the behavior actually seen.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Corrective-action knowledge base, keyed by behavior class.
#: Text is plainer for the assistant to re-word into natural language.
CORRECTIVE_ACTIONS: Dict[str, dict] = {
    "product_dropped": {
        "immediate_action": "Stop and inspect the dropped item; isolate damaged "
                            "goods and report the incident to the shift lead.",
        "process_fix": "Carry one load per trip, keep it close to the body and "
                       "at waist height, and never rush the final placement.",
        "training_unit": "Safe lifting and carrying 101",
        "category": "handling",
    },
    "product_dragged": {
        "immediate_action": "Stop the drag immediately and examine the product "
                            "for surface damage.",
        "process_fix": "Use a pallet jack or cart instead of dragging; keep "
                       "heavy items off the floor during transport.",
        "training_unit": "Load transport and pallet jacks",
        "category": "handling",
    },
    "rough_handling": {
        "immediate_action": "Slow down handling; check the product for cracks "
                            "after any impact.",
        "process_fix": "Absorb impacts with two hands and controlled "
                       "deceleration; re-handle fragile items with gloves.",
        "training_unit": "Fragile goods handling techniques",
        "category": "handling",
    },
    "incorrect_stacking": {
        "immediate_action": "Correct the stack now — redistribute weight and "
                            "re-stack before it shifts.",
        "process_fix": "Follow the stacking pattern card for this SKU; keep "
                       "the base layer level and interlock cartons.",
        "training_unit": "Pallet stacking patterns and stability",
        "category": "stacking",
    },
    "unstable_stacking": {
        "immediate_action": "Do not move the pallet; secure it with bands or "
                            "re-stack before transport.",
        "process_fix": "Scan for a stable load every layer; tall light boxes "
                       "need edge protection and a stable footprint.",
        "training_unit": "Load securing and pallet stability",
        "category": "stacking",
    },
    "outside_designated_zone": {
        "immediate_action": "Return the item to its designated bay and mark "
                            "the zone boundary visibly.",
        "process_fix": "Park loads only inside painted bay lines; free the "
                       "aisle for forklifts and pedestrians.",
        "training_unit": "Warehouse layout and zone discipline",
        "category": "layout",
    },
    "no_required_equipment": {
        "immediate_action": "Stop the task until the required equipment is "
                            "available; do not improvise.",
        "process_fix": "Bin the PPE/equipment at the station so it is grabbed "
                       "with the pick list every time.",
        "training_unit": "PPE and equipment compliance",
        "category": "safety",
    },
    "pallet_mispositioned": {
        "immediate_action": "Realign the pallet square with the rack before "
                            "loading onto it.",
        "process_fix": "Align the pallet before the first box is placed and "
                       "re-check alignment each layer.",
        "training_unit": "Rack and pallet alignment",
        "category": "stacking",
    },
    "material_pushed_or_thrown": {
        "immediate_action": "Stop throwing/pushing — retrieve the item and "
                            "inspect it for impact damage.",
        "process_fix": "Hand-carry or place, never launch; use mechanical aid "
                       "for heavy or awkward loads.",
        "training_unit": "Material handling etiquette",
        "category": "handling",
    },
    "unsafe_loading_sequence": {
        "immediate_action": "Pause the loading line and re-run the approved "
                            "sequence from the start.",
        "process_fix": "Follow the bay's loading choreography: approach, lift, "
                       "transit, place — one step at a time, never combined.",
        "training_unit": "Loading sequence and choreography",
        "category": "loading",
    },
}

#: Behavior -> suggested recurrence check (for the recurring-behavior endpoint).
TRAINING_CATEGORIES = sorted({v["category"] for v in CORRECTIVE_ACTIONS.values()})


def recommend_for(behavior_class: str, risk_level: str = "medium") -> Optional[dict]:
    """Best corrective action for a behavior, gated by risk level.

    Always returns a structured dict with a human-readable recommendation even
    for low risk; ``auto_action`` is True only at high/critical severity.
    """
    base = CORRECTIVE_ACTIONS.get(behavior_class)
    if base is None:
        return None
    auto = risk_level in ("high", "critical")
    return {
        "behavior_class": behavior_class,
        "risk_level": risk_level,
        "immediate_action": base["immediate_action"],
        "process_fix": base["process_fix"],
        "training_unit": base["training_unit"],
        "category": base["category"],
        "auto_action": auto,
        "auto_note": ("Supervisor escalation: corrective action required now."
                      if auto else
                      "Monitor; reinforce technique on the next pass."),
    }


def all_recommendations() -> List[dict]:
    return [
        recommend_for(cls, "high") for cls in CORRECTIVE_ACTIONS
    ]