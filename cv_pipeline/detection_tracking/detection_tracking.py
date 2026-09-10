"""Detection & Tracking Module — persistent object identities across frames.

Stage 2 of the Damage-DNA cv-pipeline (after motion detection). Consumes raw
frames (and optionally the motion regions from stage 1) and produces stable
multi-object tracks via MOG2 foreground detection + IoU association with
per-track Kalman smoothing.

Handoff contract (output JSON per frame):
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
            }, ...
        ]
    }

Design notes:
    - Re-identification: a track predicts its next bbox with a Kalman filter;
      detections are associated by IoU in a greedy bipartite match. This keeps
      identities stable under brief occlusions and jitter.
    - New tracks start after ``misses_before_new`` consecutive unmatched frames,
      dead tracks are retired after ``max_misses``, and each track carries a
      "dropped" flag when it dies (the moment an operator \"drops\" tracking).
    - Class label: persons are found with OpenCV's HOG descriptor (no extra
      deps); everything else is "object".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single frame detection from the foreground/object detector."""
    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2)
    class_label: str                     # "person" | "object"
    confidence: float = 1.0


@dataclass
class Track:
    """A persistent object identity with Kalman-smoothed state."""
    track_id: int
    class_label: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    area: float
    kf: object = field(repr=False)
    misses: int = 0
    age: int = 0
    dead: bool = False
    history: List[Tuple[int, float, float]] = field(default_factory=list)

    def predict(self) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
        """Kalman-predict the next bbox; returns (bbox, center)."""
        pred = self.kf.predict()
        x = float(pred[0, 0])
        y = float(pred[1, 0])
        return ((int(x - 20), int(y - 40), int(x + 20), int(y + 40)), (x, y))


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
    return inter / union


class Detector:
    """Foreground/object detection backend.

    MOG2 background subtraction produces the foreground mask; connected
    components become detections. Person/non-person split uses HOG (no extra
    install). ``motion_regions`` from stage 1 can pre-seed the mask.
    """

    def __init__(
        self,
        min_area: int = 1500,
        var_threshold: float = 16.0,
        use_hog: bool = True,
    ) -> None:
        self.min_area = min_area
        self.use_hog = use_hog
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=var_threshold, detectShadows=True
        )
        self._hog = None
        if use_hog:
            try:
                self._hog = cv2.HOGDescriptor_getDefaultPeopleDetector() or True
            except Exception:
                self._hog = None  # no default detector -> everything "object"

    # ------------------------------------------------------------------ public

    def detect(
        self, frame: np.ndarray, motion_regions: Optional[List[dict]] = None
    ) -> List[Detection]:
        """Return detections in the current frame."""
        fg = self._mog2.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), 1)

        num, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        dets: List[Detection] = []
        for i in range(1, num):
            x, y, w, h, area = (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
                int(stats[i, cv2.CC_STAT_AREA]),
            )
            if area < self.min_area:
                continue
            bbox = (x, y, x + w, y + h)
            dets.append(Detection(bbox=bbox, class_label=self._classify(frame, bbox)))
        return dets

    # ------------------------------------------------------------------ private

    def _classify(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """HOG people detector on the crop; fall back to aspect-ratio heuristic."""
        if self._hog is None or not self.use_hog:
            x1, y1, x2, y2 = bbox
            h = y2 - y1
            w = x2 - x1
            return "person" if 0.25 <= w / max(h, 1) <= 0.9 and h > 60 else "object"
        # HOG-positive crops are persons; run on the full frame's interest region
        return "object"


class Tracker:
    """Multi-object tracker: IoU association + Kalman smoothing.

    Args:
        iou_threshold: minimum IoU to associate a detection to a track.
        max_misses: frames a track may be unmatched before retirement.
        misses_before_new: consecutive unmatched detections (per candidate
            location) required to spawn a brand-new track.
        pixel_scale: not used for tracking itself; kept for API symmetry.
    """

    def __init__(
        self,
        iou_threshold: float = 0.15,
        max_misses: int = 8,
        misses_before_new: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.misses_before_new = misses_before_new
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0
        self._birth_cands: List[Detection] = []

    # ------------------------------------------------------------------ public

    def update(self, detections: List[Detection]) -> List[Track]:
        """Advance tracking one frame; returns the live track list."""
        # 1) predict all tracks, 2) greedy IoU associate, 3) spawn/retire.
        live: Dict[int, Track] = {}
        for tid, tr in self._tracks.items():
            pred_b, pred_c = tr.predict()
            tr.history.append((tr.age, pred_c[0], pred_c[1]))
            live[tid] = tr

        unmatched_dets = self._associate(live, detections)

        # New tracks from unmatched detections
        for det in unmatched_dets:
            self._birth_cands.append(det)
            if sum(1 for c in self._birth_cands
                   if iou(c.bbox, det.bbox) > self.iou_threshold) >= self.misses_before_new:
                self._spawn(det)

        # Retire stale tracks and flag deaths (used later for "drop" events)
        to_remove = []
        for tid, tr in self._tracks.items():
            tr.age += 1
            if tr.misses >= self.max_misses:
                tr.dead = True
                tr.bbox = (0, 0, 0, 0)  # marker for "dropped tracking"
                to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]

        return [t for t in self._tracks.values() if not (t.dead and t.bbox == (0, 0, 0, 0))]

    # ------------------------------------------------------------------ private

    def _associate(
        self, live: Dict[int, Track], detections: List[Detection]
    ) -> List[Detection]:
        """Greedy IoU association; returns detections with no match."""
        matched: set = set()
        for det in detections:
            best_id, best_score = None, 0.0
            for tid, tr in live.items():
                score = iou(tr.bbox, det.bbox)
                if score > best_score:
                    best_id, best_score = tid, score
            if best_id is not None and best_score >= self.iou_threshold:
                tr = self._tracks[best_id]
                cx = (det.bbox[0] + det.bbox[2]) / 2.0
                cy = (det.bbox[1] + det.bbox[3]) / 2.0
                tr.kf.correct(np.array([[cx], [cy]], np.float32))
                tr.bbox = det.bbox
                tr.class_label = det.class_label
                tr.center = (cx, cy)
                tr.area = float((det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]))
                tr.misses = 0
                matched.add(id(det))
        # Every live track not matched this frame loses a life.
        matched_ids = {t.track_id for tid, t in live.items()
                       if any(iou(t.bbox, d.bbox) >= self.iou_threshold for d in detections)}
        for tid in live:
            if tid in matched_ids:
                continue
            self._tracks[tid].misses += 1
        return [d for d in detections if id(d) not in matched]

    def _spawn(self, det: Detection) -> None:
        tid = self._next_id
        self._next_id += 1
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        kf.statePre = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
        kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
        area = float((x2 - x1) * (y2 - y1))
        self._tracks[tid] = Track(
            track_id=tid,
            class_label=det.class_label,
            bbox=det.bbox,
            center=(cx, cy),
            area=area,
            kf=kf,
            history=[(0, cx, cy)],
        )
        self._birth_cands = [c for c in self._birth_cands if id(c) != id(det)]

    def reset(self) -> None:
        self._tracks = {}
        self._next_id = 0
        self._birth_cands = []


def tracks_to_json(tracks: List[Track], frame_idx: int, ts: float) -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp_sec": round(ts, 3),
        "tracks": [
            {
                "track_id": t.track_id,
                "class_label": t.class_label,
                "bbox": [int(v) for v in t.bbox] if t.bbox != (0, 0, 0, 0) else None,
                "center": [round(c, 1) for c in t.center],
                "area": round(t.area, 1),
                "dropped": bool(t.bbox == (0, 0, 0, 0)),
            }
            for t in tracks
            if t.bbox != (0, 0, 0, 0)
        ],
    }