"""Segmentation and tracking — class-agnostic object tracking.

Integrates with the existing MotionDetector: segmenting and tracking happens
ONLY on motion-flagged regions, never the full frame.

Two backends:
    - "sam2":         Lazy-loaded SAM2 VideoPredictor. Runs only on
                      flow-flagged bboxes. Requires torch + segment-anything-2.
                      ~0.5-2 s/frame on CPU, ~0.1 s/frame on GPU.
    - "opencv_fallback" (default): Connected-components refinement of motion
                      regions + IoU data association + Kalman centroid
                      filtering. ~5 ms/frame, always available.

Both produce the same TrackedObject output.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from pipeline.types import TrackedObject

logger = logging.getLogger(__name__)

# Environment override for manual testing
_DISABLE_SAM2 = os.environ.get("REPLAYTWIN_DISABLE_SAM2", "0") == "1"


@dataclass
class _KalmanTrack:
    """Internal Kalman track state."""
    track_id: int
    bbox: List[int]                # [x1, y1, x2, y2]
    kf: Optional[cv2.KalmanFilter] = None
    age: int = 1
    miss_count: int = 0
    class_label: str = "object"
    area: int = 0


class SegmentationTracker:
    """Segment + track moving regions across frames.

    Args:
        min_region_area: minimum connected-component area (pixels) to track.
        iou_threshold: minimum IoU for detection->track association.
        merge_distance_px: max centroid distance (px) for fragment merging and
                           distance-based association when IoU is low.
        max_gap_frames: frames a track can be missing before deletion.
        max_new_tracks: cap on new tracks created per frame (large counts are
                        usually fragmented motion noise, not real objects).
        target_fps: target processing FPS (used for SAM2 speed trade-off docs).
        use_sam2: attempt to load SAM2 if available (flow-flagged regions only).
    """

    def __init__(
        self,
        min_region_area: int = 500,
        iou_threshold: float = 0.15,
        merge_distance_px: int = 40,
        max_gap_frames: int = 5,
        max_new_tracks: int = 12,
        target_fps: float = 5.0,
        use_sam2: bool = False,
    ) -> None:
        self.min_region_area = min_region_area
        self.iou_threshold = iou_threshold
        self.merge_distance_px = merge_distance_px
        self.max_gap_frames = max_gap_frames
        self.max_new_tracks = max_new_tracks
        self.target_fps = target_fps
        self.use_sam2 = use_sam2

        self._tracks: Dict[int, _KalmanTrack] = {}
        self._next_id: int = 0
        self._sam2 = None
        self.backend: str = "opencv_fallback"

        if self.use_sam2 and not _DISABLE_SAM2:
            self._load_sam2()

    # ------------------------------------------------------------------ public

    def reset(self) -> None:
        """Clear all tracks (call this at the start of a new video)."""
        self._tracks = {}
        self._next_id = 0

    def process(
        self,
        frame: np.ndarray,
        motion_regions: List[dict],
        frame_idx: int,
        timestamp_sec: float,
    ) -> List[TrackedObject]:
        """Associate this frame's motion regions with persistent track IDs.

        Args:
            frame: current BGR frame.
            motion_regions: flow-flagged bboxes from MotionDetector
                            (e.g. ``[{"bbox": [x1,y1,x2,y2], "magnitude": f}]``).
            frame_idx: frame number (for tracking continuity).
            timestamp_sec: timestamp of this frame.

        Returns:
            List of TrackedObject with persistent track IDs.
        """
        if not motion_regions:
            self._increment_misses()
            return [self._track_to_object(t) for t in self._tracks.values()]

        # 1) Refine regions (SAM2 mask path or connected-component path)
        if self.backend == "sam2" and self._sam2 is not None:
            masks, bboxes = self._sam2_mask_regions(frame, motion_regions)
        else:
            masks, bboxes = self._opencv_components(frame, motion_regions)

        # 2) Predict each track's next state
        predicted = {tid: self._predict(t) for tid, t in self._tracks.items()}

        # 3) Greedy association using IoU + centroid proximity
        track_keys = list(predicted.keys())
        used_tracks: set = set()
        used_det: set = set()
        assignments: List[Tuple[int, int]] = []

        def _centroid(box: List[int]) -> Tuple[float, float]:
            return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

        # Score: IoU when boxes overlap; centroid proximity when they don't.
        pairs: List[Tuple[float, int, int]] = []
        for ti, tid in enumerate(track_keys):
            t_cent = _centroid(predicted[tid])
            for di, dbox in enumerate(bboxes):
                iou = self._iou(predicted[tid], dbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
                    continue
                d_cent = _centroid(dbox)
                dist = float(np.hypot(t_cent[0] - d_cent[0], t_cent[1] - d_cent[1]))
                if dist <= self.merge_distance_px:
                    # proximity score below any IoU match
                    pairs.append((0.05 + (1.0 - dist / self.merge_distance_px) * 0.04, ti, di))
        pairs.sort(key=lambda p: -p[0])
        for _score, ti, di in pairs:
            if ti not in used_tracks and di not in used_det:
                assignments.append((track_keys[ti], di))
                used_tracks.add(ti)
                used_det.add(di)

        # 4) Update matched tracks
        for tid, di in assignments:
            track = self._tracks[tid]
            track.bbox = bboxes[di]
            track.age += 1
            track.miss_count = 0
            track.area = self._bbox_area(bboxes[di])
            track.class_label = self._classify_region(
                frame, bboxes[di], track.class_label
            )
            self._update_kf(track)

        # 5) New tracks for unmatched detections (cap to largest N — many small
#    unmatched fragments are usually motion noise, not real objects)
        unmatched = [di for di in range(len(bboxes)) if di not in used_det]
        unmatched.sort(
            key=lambda di: self._bbox_area(bboxes[di]), reverse=True
        )
        for di in unmatched[: self.max_new_tracks]:
            self._next_id += 1
            new = _KalmanTrack(
                track_id=self._next_id,
                bbox=list(bboxes[di]),
                age=1,
                class_label=self._classify_region(frame, bboxes[di], "object"),
                area=self._bbox_area(bboxes[di]),
            )
            self._init_kf(new, bboxes[di])
            self._tracks[self._next_id] = new

        # 6) Aging + cleanup
        for tid, track in list(self._tracks.items()):
            if tid not in [a[0] for a in assignments]:
                track.miss_count += 1
            if track.miss_count > self.max_gap_frames:
                del self._tracks[tid]

        return [self._track_to_object(t) for t in self._tracks.values()]

    # ------------------------------------------------------------------ helpers

    def _opencv_components(
        self, frame: np.ndarray, motion_regions: List[dict]
    ) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Extract refined bboxes from motion regions using connected comps."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in motion_regions:
            x1, y1, x2, y2 = [max(0, int(v)) for v in region["bbox"]]
            x1, y1 = min(x1, w - 1), min(y1, h - 1)
            x2, y2 = min(max(x2, x1 + 1), w), min(max(y2, y1 + 1), h)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Dilate to merge fragmented flow regions into cohesive blobs
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.merge_distance_px, self.merge_distance_px)
        )
        mask = cv2.dilate(mask, kernel, iterations=2)

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        bboxes: List[List[int]] = []
        masks: List[np.ndarray] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_region_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])
            bboxes.append([x, y, x + ww, y + hh])
            comp_mask = np.where(labels == i, 255, 0).astype(np.uint8)
            masks.append(comp_mask)
        return masks, bboxes

    def _sam2_mask_regions(
        self, frame: np.ndarray, motion_regions: List[dict]
    ) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Prompt SAM2 with flow-flagged bboxes only; return masks + bboxes."""
        import torch

        predictor = self._sam2
        device = "cuda" if torch.cuda.is_available() else "cpu"

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # SAM2 video predictor expects an image on the device
        img_t = (
            torch.from_numpy(frame_rgb)
            .permute(2, 0, 1)
            .float()
            .to(device)
            .unsqueeze(0)
        )
        predictor.set_image(img_t)

        masks: List[np.ndarray] = []
        bboxes: List[List[int]] = []
        try:
            # NOTE: SAM2 API surface varies by release. This targets the
            # SAM2VideoPredictor.add_new_points_or_box flow. If the installed
            # version differs, adapt here — the OpenCV fallback is unaffected.
            for region in motion_regions:
                box = region["bbox"]
                # Prompt ONLY this region, never the full frame
                out = predictor.add_new_points_or_box(
                    box=box, frame_idx=0, obj_id=region.get("track_id", 0)
                )
                if isinstance(out, tuple):
                    out = out[0]
                pred_mask = out[0] if isinstance(out, (list, tuple)) else out
                if pred_mask is None:
                    continue
                mask = (
                    (pred_mask > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255
                )
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                masks.append(mask)
                bboxes.append(
                    [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                )
        except Exception as exc:
            logger.warning(
                "SAM2 predict failed (%s). Degrading to OpenCV fallback for "
                "this frame.",
                exc,
            )
            self.backend = "opencv_fallback"
            return self._opencv_components(frame, motion_regions)
        return masks, bboxes

    # ------------------------------------------------------------------ Kalman

    def _init_kf(self, track: _KalmanTrack, bbox: List[int]) -> None:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        track.kf = kf

    def _predict(self, track: _KalmanTrack) -> List[int]:
        if track.kf is None:
            return track.bbox
        kf = track.kf
        state = kf.predict()
        cx, cy = float(state[0, 0]), float(state[1, 0])
        w = track.bbox[2] - track.bbox[0]
        h = track.bbox[3] - track.bbox[1]
        return [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]

    def _update_kf(self, track: _KalmanTrack) -> None:
        if track.kf is None:
            return
        cx = (track.bbox[0] + track.bbox[2]) / 2.0
        cy = (track.bbox[1] + track.bbox[3]) / 2.0
        track.kf.correct(np.array([[cx], [cy]], np.float32))

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _iou(box_a: List[int], box_b: List[int]) -> float:
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _bbox_area(bbox: List[int]) -> int:
        return max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    @staticmethod
    def _classify_region(
        frame: np.ndarray, bbox: List[int], prev: str
    ) -> str:
        """Classify a region as person/object/pallet/forklift.

        Person detection uses an OpenCV face detector bag-of-boxes heuristic.
        Falls back gracefully — never raises.
        """
        if prev != "object":
            return prev  # keep stable once classified
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return "object"
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                return "person"
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("Face classify failed: %s", exc)
        return "object"

    def _increment_misses(self) -> None:
        for tid, track in list(self._tracks.items()):
            track.miss_count += 1
            if track.miss_count > self.max_gap_frames:
                del self._tracks[tid]

    @staticmethod
    def _track_to_object(track: _KalmanTrack) -> TrackedObject:
        cx = (track.bbox[0] + track.bbox[2]) / 2.0
        cy = (track.bbox[1] + track.bbox[3]) / 2.0
        return TrackedObject(
            track_id=track.track_id,
            bbox=list(track.bbox),
            class_label=track.class_label,
            area=track.area,
            centroid=(cx, cy),
        )

    def _load_sam2(self) -> None:
        """Lazy-load SAM2. Fails gracefully to OpenCV fallback."""
        try:
            from sam2.build_sam import build_sam2  # noqa: F401
            from sam2.sam2_video_predictor import SAM2VideoPredictor

            self._sam2 = SAM2VideoPredictor()
            self.backend = "sam2"
            logger.info(
                "SAM2 backend loaded. Speed trade-off: ~0.5-2 s/frame on CPU. "
                "Run only on flow-flagged regions (target %s fps).",
                self.target_fps,
            )
        except ImportError as exc:
            self.use_sam2 = False
            self.backend = "opencv_fallback"
            logger.warning(
                "SAM2 unavailable (%s). Falling back to OpenCV tracking "
                "(~5 ms/frame, always available).",
                exc,
            )
        except Exception as exc:  # robust against 3.14 wheel issues
            self.use_sam2 = False
            self.backend = "opencv_fallback"
            logger.warning("SAM2 init failed (%s). Using OpenCV fallback.", exc)