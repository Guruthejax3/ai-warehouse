"""Motion Detection Module — first stage of the Damage-DNA CV pipeline.

Handoff contract (output JSON schema):
    {
        "frame_idx": int,
        "timestamp_sec": float,
        "motion_regions": [{"bbox": [x1, y1, x2, y2], "magnitude": float}, ...],
        "frame_motion_score": float,
        "event_type": "sustained" | "sudden_spike" | "none"
    }

event_type logic:
    - Compute a rolling mean + std of frame_motion_score over a configurable window.
    - If the current score exceeds rolling_mean + spike_std_multiplier * rolling_std,
      tag as "sudden_spike".
    - If the score is above motion_threshold but not a spike, tag as "sustained".
    - Otherwise "none".

Two backends:
    - "farneback" (default): OpenCV cv2.calcOpticalFlowFarneback — no extra deps.
    - "raft": torchvision.models.optical_flow.raft_small — lazy-loaded; raises a
      clear error if torch/torchvision are missing.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)

EventType = Literal["sustained", "sudden_spike", "none"]


@dataclass
class MotionRegion:
    bbox: List[int]  # [x1, y1, x2, y2]
    magnitude: float


@dataclass
class MotionResult:
    motion_mask: np.ndarray
    motion_regions: List[dict]
    frame_motion_score: float
    event_type: EventType


class MotionDetector:
    """Dense optical-flow motion detector with sustained / spike classification."""

    def __init__(
        self,
        method: str = "farneback",
        motion_threshold: float = 2.0,
        min_region_area: int = 500,
        spike_std_multiplier: float = 2.5,
        rolling_window: int = 15,
    ) -> None:
        if method not in {"farneback", "raft"}:
            raise ValueError(f"Unknown method {method!r}. Choose 'farneback' or 'raft'.")
        self.method = method
        self.motion_threshold = motion_threshold
        self.min_region_area = min_region_area
        self.spike_std_multiplier = spike_std_multiplier
        self.rolling_window = rolling_window

        # Rolling stats for spike detection
        self._scores: deque = deque(maxlen=rolling_window)
        self._raft_model: Optional[object] = None

        if method == "raft":
            self._load_raft()

    # ------------------------------------------------------------------ public

    def process_frame_pair(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> MotionResult:
        """Compare two consecutive frames and return motion regions + event type."""
        if prev_frame.shape != curr_frame.shape:
            raise ValueError(
                f"Frame shape mismatch: {prev_frame.shape} vs {curr_frame.shape}"
            )

        flow = self._compute_flow(prev_frame, curr_frame)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        motion_mask = (mag > self.motion_threshold).astype(np.uint8) * 255
        frame_motion_score = float(mag.mean())

        event_type = self._classify_event(frame_motion_score)

        regions = self._extract_regions(motion_mask, mag)

        return MotionResult(
            motion_mask=motion_mask,
            motion_regions=regions,
            frame_motion_score=frame_motion_score,
            event_type=event_type,
        )

    # ------------------------------------------------------------------ private

    def _compute_flow(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> np.ndarray:
        if self.method == "farneback":
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            return cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        # raft path
        return self._compute_flow_raft(prev, curr)

    def _compute_flow_raft(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        try:
            import torch
            from torchvision.models.optical_flow import raft_small
        except ImportError as exc:
            raise ImportError(
                "RAFT method requires torch and torchvision. Install with: "
                "pip install torch torchvision"
            ) from exc

        if self._raft_model is None:
            self._raft_model = raft_small(weights="Kinetics400_Weights.DEFAULT")
            self._raft_model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._raft_model = self._raft_model.to(device)

        prev_tensor = torch.from_numpy(prev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        curr_tensor = torch.from_numpy(curr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            flow = self._raft_model(curr_tensor.to(device), prev_tensor.to(device))
        return flow[0].permute(1, 2, 0).cpu().numpy()

    def _load_raft(self) -> None:
        """Lazy-load RAFT weights on first use; validates deps up-front."""
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "torch and torchvision are required for method='raft'. "
                "Install with: pip install torch torchvision"
            ) from exc
        logger.info("RAFT backend selected — model will load on first frame.")

    def _extract_regions(
        self, mask: np.ndarray, magnitude: np.ndarray
    ) -> List[dict]:
        """Find connected components in the motion mask and return their bboxes."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        regions: List[dict] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_region_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            # Mean magnitude inside this region
            region_mag = float(magnitude[labels == i].mean())
            regions.append(
                {"bbox": [x, y, x + w, y + h], "magnitude": region_mag}
            )
        return regions

    def _classify_event(self, score: float) -> EventType:
        """Classify motion as sustained / sudden_spike / none using rolling stats."""
        if len(self._scores) < 3:
            self._scores.append(score)
            return "sustained" if score > self.motion_threshold else "none"

        mean = float(np.mean(self._scores))
        std = float(np.std(self._scores))
        self._scores.append(score)

        if std > 0 and score > mean + self.spike_std_multiplier * std:
            return "sudden_spike"
        if score > self.motion_threshold:
            return "sustained"
        return "none"
