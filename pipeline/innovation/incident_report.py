"""Automatic incident report generation — structured PDF/text export.

Turns a single RiskEvent (plus optional damage prediction and corrective
action) into a machine-readable and human-readable incident report. The
report is structured as a dataclass that serialises to JSON and can be
rendered into a PDF, HTML, or plain-text template by the caller.

Privacy: the report never includes raw video frames. It references the
evidence clip path (already face-blurred and trimmed) and the structured
trajectory data only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IncidentReport:
    """Structured incident report for one event."""
    report_id: str
    generated_at: str
    event_id: str
    behavior_class: str
    risk_level: str
    risk_score: float
    confidence: Optional[float]
    zone_id: str
    source_video: str
    timestamp_sec: float
    justification: str
    evidence_clip_path: str
    activity_type: str
    # Enriched fields
    damage_prediction: Optional[dict] = None
    corrective_action: Optional[dict] = None
    sequence_steps: Optional[List[dict]] = None
    # Narrative
    executive_summary: str = ""
    root_cause: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    follow_up_required: bool = False


def generate_incident_report(
    event_id: str,
    behavior_class: str,
    risk_level: str,
    risk_score: float,
    zone_id: str,
    source_video: str = "",
    timestamp_sec: float = 0.0,
    justification: str = "",
    evidence_clip_path: str = "",
    activity_type: str = "unknown",
    confidence: Optional[float] = None,
    damage_prediction: Optional[dict] = None,
    corrective_action: Optional[dict] = None,
    sequence_steps: Optional[List[dict]] = None,
) -> IncidentReport:
    """Build a structured incident report from event context.

    All inputs come from the pipeline's structured outputs — nothing is
    invented or free-generated. The executive_summary and root_cause are
    deterministic templates filled from the structured data.

    Returns:
        IncidentReport ready for JSON serialisation or PDF rendering.
    """
    now = datetime.now(timezone.utc).isoformat()
    report_id = f"IR-{event_id[:12]}-{now[:10]}"

    # Executive summary: one-paragraph briefing for a supervisor
    severity_word = {
        "low": "minor", "medium": "moderate",
        "high": "serious", "critical": "critical",
    }.get(risk_level, "unclassified")

    conf_str = f"{confidence:.0%}" if confidence is not None else "N/A"
    executive_summary = (
        f"On {now[:10]}, an automated safety monitoring system detected a "
        f"{severity_word} safety incident in zone {zone_id}. "
        f"The detected behavior was \"{behavior_class.replace('_', ' ')}\" "
        f"with a risk score of {risk_score:.2f} (confidence: {conf_str}). "
        f"Justification: {justification}"
    )

    # Root cause: deterministic mapping from behavior to likely cause
    _ROOT_CAUSES = {
        "product_dropped": "Operator lost grip or rushed placement during manual handling.",
        "product_dragged": "Item dragged across surface instead of being lifted — "
                           "possible fatigue or lack of material handling equipment.",
        "rough_handling": "Excessive force applied to product during handling; "
                          "possible inadequate training on fragile goods protocols.",
        "incorrect_stacking": "Stacking pattern deviated from standard; possible "
                              "rushing or unfamiliarity with the SKU's stacking card.",
        "unstable_stacking": "Stack height or weight distribution creates a "
                              "topple risk; possible overloading the base layer.",
        "outside_designated_zone": "Item placed outside the painted bay boundary; "
                                    "possible unclear zone markings or overcrowding.",
        "no_required_equipment": "Required PPE or handling equipment not in use; "
                                  "possible availability or compliance gap.",
        "pallet_mispositioned": "Pallet not aligned with the rack opening; "
                                 "possible operator positioning error.",
        "material_pushed_or_thrown": "Material thrown or pushed rather than "
                                      "placed; possible time pressure.",
        "unsafe_loading_sequence": "Loading steps performed out of order or "
                                    "combined (approach-handle-transit-place); "
                                    "possible skipping rest steps.",
    }
    root_cause = _ROOT_CAUSES.get(behavior_class, "Under investigation.")

    # Recommended actions: start with corrective action, add damage-specific
    recommended_actions = []
    if corrective_action:
        recommended_actions.append(corrective_action.get("immediate_action", ""))
        recommended_actions.append(corrective_action.get("process_fix", ""))
    if damage_prediction:
        recommended_actions.append(damage_prediction.get("immediate_response", ""))
    if risk_level in ("high", "critical"):
        recommended_actions.append("Notify the shift supervisor and log in the safety register.")
    # Always recommend training follow-up
    if corrective_action and corrective_action.get("training_unit"):
        recommended_actions.append(
            f"Schedule refresher training: {corrective_action['training_unit']}."
        )

    follow_up_required = risk_level in ("high", "critical") or (
        damage_prediction and damage_prediction.get("severity_tier") in ("severe", "catastrophic")
    )

    return IncidentReport(
        report_id=report_id,
        generated_at=now,
        event_id=event_id,
        behavior_class=behavior_class,
        risk_level=risk_level,
        risk_score=risk_score,
        confidence=confidence,
        zone_id=zone_id,
        source_video=source_video,
        timestamp_sec=timestamp_sec,
        justification=justification,
        evidence_clip_path=evidence_clip_path,
        activity_type=activity_type,
        damage_prediction=damage_prediction,
        corrective_action=corrective_action,
        sequence_steps=sequence_steps,
        executive_summary=executive_summary,
        root_cause=root_cause,
        recommended_actions=[a for a in recommended_actions if a],
        follow_up_required=follow_up_required,
    )


def report_to_dict(report: IncidentReport) -> dict:
    """Serialise an IncidentReport to a JSON-safe dict."""
    return asdict(report)


def report_to_json(report: IncidentReport, indent: int = 2) -> str:
    """Serialise an IncidentReport to a JSON string."""
    return json.dumps(report_to_dict(report), indent=indent, default=str)
