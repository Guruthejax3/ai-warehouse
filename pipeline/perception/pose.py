"""Pose estimation — human keypoints for persons in the scene.

Integrates with tracking: pose runs only on person-classified TrackedObjects.

Backends:
    - "mediapipe": MediaPipe Pose (33 landmarks). Lazy-imported — if
      unavailable or the wheel fails on the host Python, degrades gracefully.
    - None: pose data skipped; trajectories fall back to bbox centers only.

Output: Dict[track_id, PoseKeypoints]
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import cv2

from pipeline.types import Keypoint, PoseKeypoints, TrackedObject

logger = logging.getLogger(__name__)


class PoseEstimator:
    """Estimate body keypoints for tracked persons.

    Args:
        enabled: if False, returns no poses (best off).
        use_mediapipe: attempt MediaPipe. If it fails to load, pose is skipped.
        min_visibility: keypoints below this visibility are zeroed.
    """

    def __init__(
        self,
        enabled: bool = True,
        use_mediapipe: bool = True,
        min_visibility: float = 0.3,
    ) -> None:
        self.enabled = enabled
        self.use_mediapipe = use_mediapipe
        self.min_visibility = min_visibility
        self._mp = None
        self._pose = None
        self.source: str = "none"

        if self.enabled and self.use_mediapipe:
            self._load_mediapipe()

    # ------------------------------------------------------------------ public

    def process(
        self,
        frame: np.ndarray,
        tracked: List[TrackedObject],
    ) -> Dict[int, PoseKeypoints]:
        """Estimate poses for tracked objects.

        Args:
            frame: current BGR frame.
            tracked: TrackedObjects from SegmentationTracker.

        Returns:
            Mapping of track_id -> PoseKeypoints for person-classified tracks.
        """
        poses: Dict[int, PoseKeypoints] = {}
        if not self.enabled or tracked is None:
            return poses

        if self._pose is None:
            logger.debug("Pose backend unavailable; skipping pose estimation.")
            return poses

        for obj in tracked:
            if obj.class_label != "person":
                continue
            x1, y1, x2, y2 = [max(0, int(v)) for v in obj.bbox]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                results = self._pose.process(rgb)
            except Exception as exc:  # defensive; never crash the pipeline
                logger.warning("Pose process error on track %s: %s", obj.track_id, exc)
                continue
            if results is None or results.pose_landmarks is None:
                continue

            kps: List[Keypoint] = []
            angles: Dict[str, float] = {}
            landmarks = results.pose_landmarks.landmark
            for lm in landmarks:
                vis = float(getattr(lm, "visibility", 1.0))
                k = Keypoint(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=vis if vis >= self.min_visibility else 0.0,
                )
                kps.append(k)

            if len(kps) >= 33:
                angles = self._compute_angles(kps)

            poses[obj.track_id] = PoseKeypoints(
                track_id=obj.track_id,
                keypoints=kps,
                bbox=obj.bbox,
                angles=angles,
                source=self.source,
            )
        return poses

    # ------------------------------------------------------------------ helpers

    def _load_mediapipe(self) -> None:
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False, min_detection_confidence=0.5
            )
            self.source = "mediapipe"
            logger.info("MediaPipe Pose loaded (33 landmarks).")
        except ImportError as exc:
            self._pose = None
            self.source = "none"
            logger.warning(
                "MediaPipe unavailable (%s). Pose estimation skipped. "
                "Trajectories will use bbox centers only.",
                exc,
            )
        except Exception as exc:
            self._pose = None
            self.source = "none"
            logger.warning(
                "MediaPipe init failed on this platform (%s). Pose skipped.", exc
            )

    @staticmethod
    def _compute_angles(kps: List[Keypoint]) -> Dict[str, float]:
        """Compute joint angles from MediaPipe landmark indices.

        Landmark indices (MediaPipe): 11/12 shoulders, 13/14 elbows,
        15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles.
        """
        angles: Dict[str, float] = {}

        def _angle(a, b, c):
            va = np.array([a.x, a.y, a.z])
            vb = np.array([b.x, b.y, b.z])
            vc = np.array([c.x, c.y, c.z])
            ab = va - vb
            cb = vc - vb
            denom = np.linalg.norm(ab) * np.linalg.norm(cb)
            if denom == 0:
                return 0.0
            cos = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
            return float(np.degrees(np.arccos(cos)))

        if len(kps) >= 28:
            left_elbow = _angle(kps[11], kps[13], kps[15])
            right_elbow = _angle(kps[12], kps[14], kps[16])
            left_knee = _angle(kps[23], kps[25], kps[27])
            right_knee = _angle(kps[24], kps[26], kps[28])
            angles["left_elbow"] = left_elbow
            angles["right_elbow"] = right_elbow
            angles["left_knee"] = left_knee
            angles["right_knee"] = right_knee

            # Trunk lean: angle between shoulders line, hips line and vertical
            shoulder_mid = np.array(
                [(kps[11].x + kps[12].x) / 2, (kps[11].y + kps[12].y) / 2]
            )
            hip_mid = np.array(
                [(kps[23].x + kps[24].x) / 2, (kps[23].y + kps[24].y) / 2]
            )
            vertical = np.array([0.0, 1.0])
            trunk = hip_mid - shoulder_mid
            denom = np.linalg.norm(trunk)
            if denom > 0:
                lean = np.degrees(
                    np.arccos(np.clip(np.dot(trunk, vertical) / denom, -1.0, 1.0))
                )
                angles["trunk_lean"] = lean
        return angles

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None and hasattr(self._pose, "close"):
            try:
                self._pose.close()
            except Exception:  # pragma: no cover
                pass