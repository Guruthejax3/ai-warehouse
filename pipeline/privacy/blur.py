"""Privacy guard — evidence clips only, faces blurred, raw footage never kept.

Policy enforced here:
    1. Evidence clips are trimmed to ±5 seconds around the event peak.
    2. Faces in the clip are blurred (Haar face cascade).
    3. The full raw video is never written by the pipeline — at most a tiny
       evidence clip per event is persisted to data/evidence_clips/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


@dataclass
class EvidenceClip:
    """Where the trimmed, blurred, privacy-safe clip was written."""
    path: Path
    start_sec: float
    end_sec: float
    num_faces_blurred: int = 0


def _blur_frame(frame: np.ndarray,
                faces: Optional[List[Tuple[int, int, int, int]]] = None) -> int:
    """Blur all detected faces in a BGR frame. Returns how many were blurred.

    Falls back to a light global blur if the cascade fails to load — never
    leave identifiable faces unblurred and never raise.
    """
    if not _has_cascade():
        # If we cannot detect faces, blur the whole frame rather than risk
        # leaking an identifiable person.
        h, w = frame.shape[:2]
        frame[:] = cv2.GaussianBlur(frame, (0, 0), 15.0)
        return 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))
    rects = faces if faces is not None else [
        (int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections
    ]
    for (x, y, w, h) in rects:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        roi = frame[y0:y1, x0:x1]
        if roi.size:
            frame[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (0, 0), 25.0)
    return len(rects)


def _has_cascade() -> bool:
    try:
        return Path(_FACE_CASCADE_PATH).exists()
    except Exception:
        return False


def blur_faces(frame: np.ndarray,
               faces: Optional[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
    """Return a face-blurred copy of a BGR frame (input untouched)."""
    out = frame.copy()
    _blur_frame(out, faces=faces)
    return out


def extract_evidence_clip(
    video_path: Path,
    peak_sec: float,
    output_dir: Path,
    trim_sec: float = 5.0,
    blur: bool = True,
    event_id: Optional[str] = None,
) -> EvidenceClip:
    """Trim ±trim_sec around peak_sec, blur faces, write a single clip.

    Args:
        video_path: source video (never stored itself).
        peak_sec: event peak timestamp in the video.
        output_dir: where evidence clips land (gitignored).
        trim_sec: seconds kept on each side of the peak (default 5).
        blur: blur faces (privacy default on).
        event_id: filename slug; defaults to the source stem.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / src_fps if frame_count else 0.0

    start = max(0.0, peak_sec - trim_sec)
    end = min(duration, peak_sec + trim_sec) if duration else peak_sec + trim_sec

    slug = event_id or video_path.stem
    out_path = output_dir / f"{slug}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: Optional[cv2.VideoWriter] = None
    faces_blurred = 0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pos_ms > end:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), fourcc, int(src_fps), (w, h))
        if blur:
            faces_blurred += _blur_frame(frame)
        writer.write(frame)

    if writer is not None:
        writer.release()
    cap.release()
    return EvidenceClip(
        path=out_path, start_sec=round(start, 2), end_sec=round(end, 2),
        num_faces_blurred=faces_blurred,
    )