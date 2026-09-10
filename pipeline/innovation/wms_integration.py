"""WMS/CCTV integration — webhook handlers for external system events.

Provides lightweight webhook receiver and sender interfaces for:
    1. WMS (Warehouse Management System) — receives pick/order events,
       can push safety alerts back.
    2. CCTV/VMS — receives motion/alarm triggers, can push analysis results.

Both are designed as thin wrappers that produce structured dicts; actual
HTTP transport is handled by the FastAPI backend endpoints that call these.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Shared secret for webhook signature verification (env or default demo key)
_WEBHOOK_SECRET = os.environ.get("REPLAYTWIN_WEBHOOK_SECRET", "demo-secret-key")


@dataclass
class WMSAlert:
    """Outbound safety alert to the WMS."""
    alert_id: str
    timestamp: str
    severity: str           # info | warning | critical
    zone_id: str
    behavior_class: str
    risk_score: float
    message: str
    action_required: str
    event_id: str


@dataclass
class CCTVEvent:
    """Inbound motion/alarm event from a CCTV/VMS system."""
    source_camera: str
    timestamp: str
    event_type: str         # motion | intrusion | tamper | line_cross
    zone_id: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify HMAC-SHA256 webhook signature from WMS/CCTV.

    Returns True if the signature matches the shared secret.
    """
    expected = hmac.new(
        _WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def build_wms_alert(
    event_id: str,
    behavior_class: str,
    risk_level: str,
    risk_score: float,
    zone_id: str,
    justification: str = "",
) -> WMSAlert:
    """Build a structured alert for the WMS from a pipeline event.

    Maps risk levels to WMS severity:
        low/medium -> info
        high       -> warning
        critical   -> critical
    """
    severity_map = {"low": "info", "medium": "info", "high": "warning", "critical": "critical"}
    action_map = {
        "info": "Log and monitor.",
        "warning": "Notify supervisor; consider pausing operations in zone.",
        "critical": "STOP operations in zone; dispatch safety team immediately.",
    }
    severity = severity_map.get(risk_level, "info")
    now = datetime.now(timezone.utc).isoformat()

    return WMSAlert(
        alert_id=f"WA-{event_id[:12]}",
        timestamp=now,
        severity=severity,
        zone_id=zone_id,
        behavior_class=behavior_class,
        risk_score=risk_score,
        message=(
            f"Safety alert [{severity.upper()}]: {behavior_class.replace('_', ' ')} "
            f"detected in {zone_id} (risk: {risk_score:.2f}). {justification}"
        ),
        action_required=action_map[severity],
        event_id=event_id,
    )


def parse_cctv_webhook(payload: dict) -> CCTVEvent:
    """Parse an incoming CCTV/VMS webhook payload.

    Accepts common CCTV event formats and normalises to CCTVEvent.
    Unknown fields go into metadata.
    """
    known_keys = {"source_camera", "timestamp", "event_type", "zone_id", "confidence"}
    metadata = {k: v for k, v in payload.items() if k not in known_keys}

    return CCTVEvent(
        source_camera=payload.get("source_camera", "unknown"),
        timestamp=payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
        event_type=payload.get("event_type", "motion"),
        zone_id=payload.get("zone_id", "bay_0"),
        confidence=float(payload.get("confidence", 0.5)),
        metadata=metadata,
    )


def alert_to_dict(alert: WMSAlert) -> dict:
    return asdict(alert)


def cctv_to_dict(event: CCTVEvent) -> dict:
    return asdict(event)
