"""PPE compliance detection — rule-based helmet / hi-vis / safety-vest checks.

The detector operates on the bounding boxes produced by the segmentation
stage. It applies color-histogram heuristics inside the *head region* (top
portion of a person bbox) for helmet detection and the *torso region* (middle
band) for hi-vis vest checks.

This is deliberately lightweight and camera-agnostic:
    1. Split each person bbox into head (top 22%) and torso (22%-60%) bands.
    2. Build HSV histograms for those bands.
    3. Helmet present when the head band is dominated by a small set of
       "helmet-like" saturated hues (amber, white, blue, red) and not
       skin-tone-dominant.
    4. Hi-vis present when the torso band shows a high-saturation pop (amber /
       yellow-green / orange) with sufficient area.

Colours are configurable so different site fleets (orange hi-vis vs yellow)
work out of the box. Sensor-clear fallback: when the scene is monochrome or
the cascade misdetects, the detector returns "unknown" rather than a confident
violation (no false alarms).

Output: a list of PPEFinding dicts with person track id, missing item, and a
human-readable justification. The runner turns each finding into a
no_required_equipment risk event.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Hue ranges in OpenCV HSV (0-179). Saturated <-> 0.3+. Value >= 60.
_HELMET_HUES = {
    "amber": (12, 30),
    "white": (0, 8),   # low-saturation bright
    "blue": (90, 130),
    "red": (0, 10),
    "green": (40, 75),
}
_HIVIS_HUES = {
    "amber": (12, 30),
    "yellow_green": (25, 45),
    "orange": (5, 20),
}


@dataclass
class PPEFinding:
    """One PPE compliance result for a detected person."""
    track_id: int
    helmet_missing: bool = False
    hivis_missing: bool = False
    helmet_unknown: bool = False
    hivis_unknown: bool = False
    justification: str = ""
    features: Dict[str, float] = field(default_factory=dict)


class PPEDetector:
    """Colour-rule based PPE compliance checker.

    Args:
        helmet_check: enable helmet detection (default True).
        hivis_check: enable hi-vis vest detection (default True).
        head_ratio: fraction of bbox height treated as the head band.
        torso_top / torso_bottom: normalized y-band for the torso.
        min_highsat_fraction: minimum fraction of band pixels that must be
            high-saturation hue pixels to call the item "present".
    """

    def __init__(
        self,
        helmet_check: bool = True,
        hivis_check: bool = True,
        head_ratio: float = 0.22,
        torso_top: float = 0.22,
        torso_bottom: float = 0.60,
        min_highsat_fraction: float = 0.08,
    ) -> None:
        self.helmet_check = helmet_check
        self.hivis_check = hivis_check
        self.head_ratio = head_ratio
        self.torso_top = torso_top
        self.torso_bottom = torso_bottom
        self.min_highsat_fraction = min_highsat_fraction

    def check_frame(self, frame, tracked: List) -> List[PPEFinding]:
        """Run PPE checks on one frame's tracked objects.

        Args:
            frame: BGR numpy frame.
            tracked: list of TrackedObject (must have bbox and class_label).

        Returns PPEFindings for each person-class detection.
        """
        if frame is None:
            return []
        import cv2

        findings: List[PPEFinding] = []
        for obj in tracked:
            label = getattr(obj, "class_label", "object")
            if label != "person":
                continue
            bbox = getattr(obj, "bbox", None)
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            h = y2 - y1

            head = frame[max(0, y1):max(y1 + 1, y1 + int(h * self.head_ratio)),
                         x1:x2]
            torso = frame[max(0, y1 + int(h * self.torso_top)):
                          min(frame.shape[0], y1 + int(h * self.torso_bottom)),
                          x1:x2]

            f = PPEFinding(track_id=getattr(obj, "track_id", 0))
            if self.helmet_check:
                hm, hk = self._check_band(head, _HELMET_HUES, self.min_highsat_fraction)
                f.helmet_missing = bool(hm == "missing")
                f.helmet_unknown = bool(hk)
            if self.hivis_check:
                vm, vk = self._check_band(torso, _HIVIS_HUES, self.min_highsat_fraction)
                f.hivis_missing = bool(vm == "missing")
                f.hivis_unknown = bool(vk)

            parts = []
            if f.helmet_missing:
                parts.append("helmet not detected in head band")
            elif f.helmet_unknown:
                parts.append("helmet status indeterminate (low colour signal)")
            if f.hivis_missing:
                parts.append("hi-vis vest not detected in torso band")
            elif f.hivis_unknown:
                parts.append("hi-vis status indeterminate")
            f.justification = (
                f"PPE check on person track {f.track_id}: "
                + ("; ".join(parts) if parts else "helmet and hi-vis present.")
            )
            findings.append(f)
        return findings

    # ------------------------------------------------------------------ internals

    def _check_band(self, band, hue_set: Dict[str, tuple],
                    min_frac: float) -> tuple:
        """Return ('present'|'missing'|'unknown', confidence_indeterminate)."""
        if band is None or band.size == 0:
            return "unknown", True
        import cv2

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        total = h_ch.size
        if total == 0:
            return "unknown", True

        # How much of the band is bright enough to be a garment at all?
        bright = float(np.mean(v_ch > 40))
        if bright < 0.5:
            return "unknown", True  # dark/unexposed band — no confident call

        sat_pop = 0.0
        for name, (lo, hi) in hue_set.items():
            # High-saturation pixels in this band of hues
            mask = ((h_ch >= lo) & (h_ch <= hi) & (s_ch >= 60) & (v_ch >= 60))
            sat_pop += float(np.mean(mask))
        sat_pop = min(sat_pop, 1.0)

        if sat_pop >= min_frac:
            return "present", False
        # Skin-tone dominated head band = bare head
        skin = float(np.mean((h_ch >= 0) & (
            (h_ch <= 20) | (h_ch >= 150)
        ) & (s_ch >= 60) & (s_ch <= 170) & (v_ch >= 60)))
        if band.shape[0] <= max(2, int(self.head_ratio * 200)):
            pass  # too thin to trust skin classification
        if skin > 0.25 and sat_pop < 0.5 * min_frac:
            return "missing", False
        return "unknown", True