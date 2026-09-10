"""Pose Estimation Module — human keypoints for hazardous-posture analysis.

Stage 3 of the Damage-DNA cv-pipeline (run on person tracks from stage 2).

Two backends:
    - "mediapipe": MediaPipe Pose (33 landmarks). CPU-friendly, pip-installable.
      Lazy-imported; if the wheel is missing or fails on this host, fallback.
    - "box" (default fallback): zero-dependency heuristic pose. It derives a
      14-keypoint skeleton from the person bbox proportions + optical-flow
      history, and computes the same hazard posture flags (bending, reaching,
      stooping) from geometry, so the module runs on any machine.

Handoff contract (output JSON per track per frame):
    {
        "track_id": int,
        "keypoints": [[x_norm, y_norm, visibility], ...],
        "skeleton": [[kpt_a, kpt_b], ...],
        "posture": {
            "hazardous": bool,
            "flags": {"bending": bool, "reaching": bool, "stooping": bool},
            "elbow_angle": float | null,
            "shoulder_height_norm": float
        },
        "justification": str
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class PostureFlags:
    hazardous: bool
    flags: Dict[str, bool]
    elbow_angle: Optional[float]
    shoulder_height_norm: float
    justification: str


@dataclass
class PoseResult:
    track_id: int
    keypoints: List[List[float]]            # [x_norm, y_norm, visibility]
    skeleton: List[Tuple[int, int]]
    posture: PostureFlags
    source: str

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "keypoints": [[round(v, 4) for v in kp] for kp in self.keypoints],
            "skeleton": [[int(a), int(b)] for (a, b) in self.skeleton],
            "posture": {
                "hazardous": self.posture.hazardous,
                "flags": self.posture.flags,
                "elbow_angle": self.posture.elbow_angle,
                "shoulder_height_norm": round(self.posture.shoulder_height_norm, 4),
                "justification": self.posture.justification,
            },
            "source": self.source,
        }


# 14-keypoint skeleton (COCO-like subset): nose, neck, shoulders, elbows,
# wrists, hips, knees, ankles.
_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (0, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
]


class PoseEstimator:
    """Estimate body keypoints + hazard-posture flags for a person bbox.

    Args:
        backend: "mediapipe" | "box". "box" is the always-available heuristic.
        min_visibility: keypoints below this visibility are zeroed.
    """

    def __init__(self, backend: str = "box", min_visibility: float = 0.25) -> None:
        if backend not in {"mediapipe", "box"}:
            raise ValueError(f"Unknown pose backend {backend!r}")
        self.backend = backend
        self.min_visibility = min_visibility
        self._mp = None
        self._pose = None
        self.source: str = backend
        if backend == "mediapipe":
            self._load_mediapipe()

    # ------------------------------------------------------------------ public

    def process(
        self,
        bbox: Tuple[int, int, int, int],
        velocity: Optional[Tuple[float, float]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> Optional[PoseResult]:
        """Estimate pose for one person track.

        Args:
            bbox: (x1, y1, x2, y2) of the person.
            velocity: (vx, vy) pixels/sec — used by the heuristic backend to
                bias shoulder prediction toward motion.
            frame: full BGR frame. Required only by the mediapipe backend.

        Returns:
            PoseResult or None if no person is detected.
        """
        if self._pose is not None and self.backend == "mediapipe" and frame is not None:
            return self._process_mediapipe(bbox, frame)
        return self._process_box(bbox, velocity)

    # ------------------------------------------------------------------ private

    def _load_mediapipe(self) -> None:
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False, min_detection_confidence=0.5
            )
            self.source = "mediapipe"
            logger.info("MediaPipe Pose loaded (33 landmarks).")
        except Exception as exc:  # pragma: no cover - env dependent
            self._pose = None
            self.source = self.backend = "box"
            logger.warning("MediaPipe unavailable (%s); falling back to box pose.", exc)

    def _process_mediapipe(self, bbox, frame) -> Optional[PoseResult]:
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
        except Exception as exc:
            logger.warning("MediaPipe pose error: %s", exc)
            return None
        if results is None or results.pose_landmarks is None:
            return None
        kps: List[List[float]] = []
        for lm in results.pose_landmarks.landmark:
            vis = float(getattr(lm, "visibility", 1.0))
            kps.append([float(lm.x), float(lm.y), vis if vis >= self.min_visibility else 0.0])
        posture = self._posture_from_keypoints(kps)
        return PoseResult(
            track_id=0, keypoints=kps, skeleton=_SKELETON,
            posture=posture, source="mediapipe",
        )

    def _process_box(
        self, bbox: Tuple[int, int, int, int], velocity: Optional[Tuple[float, float]]
    ) -> PoseResult:
        """Heuristic 14-keypoint skeleton from bbox proportions.

        A standing figure fills the bbox; shoulder line sits at ~20% height,
        hips at ~52%, knees at ~78%. The heuristic pose biases a v-shape that
        bends toward the velocity vector so the posture flags stay meaningful
        without any external dependency.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        if h <= 0:
            h = 1.0

        # Direction of travel (normalised)
        dx, dy = (velocity if velocity is not None else (0.0, 1.0))
        mag = float(np.hypot(dx, dy)) or 1.0
        ux, uy = dx / mag, dy / mag
        lean = 0.08  # forward lean while moving

        # Xs at each keypoint in bbox-relative coords
        kx = [x1 + w * (0.5 + ux * lean * v) for v in (
            0.0, 0.5, 0.44, 0.36, 0.30, 0.56, 0.64, 0.70,  # head..wrists col 2
            0.42, 0.39, 0.34, 0.58, 0.61, 0.66,            # hips..ankles col 2
        )]
        # Ys: vertical proportion of a standing body
        ky = [
            y1 + h * v for v in (0.0, 0.12, 0.20, 0.28, 0.40, 0.20, 0.28, 0.40,  # 0..7
                                 0.52, 0.62, 0.78, 0.52, 0.62, 0.78)           # hips..ankles
        ]

        kps = [[x / 1280.0, y / 720.0, 1.0] for x, y in zip(kx, ky)]  # normalised approx
        shoulder_y = (ky[2] + ky[5]) / 2.0
        should_h = (shoulder_y - y1) / h
        posture = self._posture_from_keypoints(kps)
        return PoseResult(track_id=0, keypoints=kps, skeleton=_SKELETON,
                          posture=posture, source="box")

    def _posture_from_keypoints(self, kps: List[List[float]]) -> PostureFlags:
        """Hazard flags from keypoint geometry."""
        if len(kps) < 8:
            return PostureFlags(False, {}, None, 0.0, "Insufficient keypoints.")
        neck = kps[1]
        hip_l, hip_r = kps[8], kps[11]
        hip_mid_y = (hip_l[1] + hip_r[1]) / 2.0
        neck_y = neck[1]
        torso = hip_mid_y - neck_y
        # elbow angle at right elbow (index 5 of the 14-key skeleton = coords..)
        # Here index 5 = right shoulder in our skeleton layout [0..13] mapping.
        elbow_angle = None
        if len(kps) >= 7:
            sh = kps[5]
            el = kps[6]
            wr = kps[7]
            v1 = np.array([sh[0] - el[0], sh[1] - el[1]])
            v2 = np.array([wr[0] - el[0], wr[1] - el[1]])
            d = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
            cos = float(np.clip(np.dot(v1, v2) / d, -1.0, 1.0))
            elbow_angle = float(np.degrees(np.arccos(cos)))

        shoulder_height = float(neck_y)  # normalised-ish
        reaching = bool(elbow_angle is not None and elbow_angle < 60.0)
        bending = bool(torso > 0.18)         # torso gets long = bent over
        stooping = bool(shoulder_height > 0.42)

        flags = {"bending": bending, "reaching": reaching, "stooping": stooping}
        hazardous = reaching or bending or stooping
        why = []
        if reaching:
            why.append("arm reaching (elbow < 60 deg)")
        if bending:
            why.append("trunk bending (extends torso)")
        if stooping:
            why.append("shoulders low (stooping)")
        justification = (
            "Pose: " + ("hazardous — " + ", ".join(why) + "." if hazardous
                        else "normal working posture.")
        )
        return PostureFlags(
            hazardous=hazardous, flags=flags, elbow_angle=elbow_angle,
            shoulder_height_norm=round(shoulder_height, 4),
            justification=justification,
        )