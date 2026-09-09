"""Behaviour matching — DTW against a JSON exemplar library.

Given a trajectory (or a set of them), return the closest exemplar behaviour
class. Default algorithm is fastdtw; a pure-numpy DTW is the fallback so the
pipeline works without fastdtw installed.

Exemplar JSON schema (data/exemplars/behaviors.json):
    {
      "exemplars": [
        {
          "id": "product_dragged_001",
          "behavior_class": "product_dragged",
          "label": "Cupboard dragged across floor",
          "points": [{"frame_idx":0,"timestamp_sec":0.0,"x":1.2,"y":3.4,
                      "z_est":0.0,"vx":0.1,"vy":0.2}, ...],
          "source_clip": null,
          "metadata": {"fragility": "standard"}
        }
      ]
    }
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline.types import BEHAVIOR_CLASSES, BehaviorMatch, Trajectory

logger = logging.getLogger(__name__)

RESAMPLE_N = 32  # fixed trajectory length for distance comparison


class ExemplarLibrary:
    """Load, query, and extend a JSON exemplar library for behaviour matching."""

    def __init__(self, path: str = "data/exemplars/behaviors.json") -> None:
        self.path = Path(path)
        self.exemplars: List[Dict[str, Any]] = []
        self.reload()

    # ------------------------------------------------------------------ load

    def reload(self) -> None:
        """(Re)load exemplars from disk."""
        if not self.path.exists():
            logger.warning("Exemplar library not found at %s — empty.", self.path)
            self.exemplars = []
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.exemplars = data.get("exemplars", [])
        except Exception as exc:
            logger.warning("Failed to load exemplar library (%s) — empty.", exc)
            self.exemplars = []

    # ------------------------------------------------------------------ query

    def all(self) -> List[Dict[str, Any]]:
        return self.exemplars

    def by_class(self, behavior_class: str) -> List[Dict[str, Any]]:
        return [e for e in self.exemplars if e.get("behavior_class") == behavior_class]

    def classes_present(self) -> List[str]:
        return sorted({e.get("behavior_class", "") for e in self.exemplars})

    # ------------------------------------------------------------------ write

    def add_exemplar(
        self,
        behavior_class: str,
        label: str,
        points: List[dict],
        source_clip: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Append an exemplar and persist to disk. Returns its id."""
        if behavior_class not in BEHAVIOR_CLASSES:
            raise ValueError(
                f"Unknown behavior class {behavior_class!r}. "
                f"Must be one of {BEHAVIOR_CLASSES}"
            )
        exemplar_id = f"{behavior_class}_{uuid.uuid4().hex[:8]}"
        entry = {
            "id": exemplar_id,
            "behavior_class": behavior_class,
            "label": label,
            "points": points,
            "source_clip": source_clip,
            "metadata": metadata or {},
        }
        self.exemplars.append(entry)
        self._persist()
        return exemplar_id

    def remove_exemplar(self, exemplar_id: str) -> bool:
        before = len(self.exemplars)
        self.exemplars = [e for e in self.exemplars if e.get("id") != exemplar_id]
        changed = len(self.exemplars) != before
        if changed:
            self._persist()
        return changed

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "exemplars": self.exemplars,
            "version": "1.0",
            "updated": __import__("datetime").datetime.now().isoformat(),
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Feature extraction + DTW
# ---------------------------------------------------------------------------

def extract_trajectory_vector(points: List[dict]) -> np.ndarray:
    """Resample a trajectory to a fixed-length feature matrix of shape
    (RESAMPLE_N, 3): channels are [x_rel_norm, y_rel_norm, speed_norm].

    Preprocessing for shape-based matching:
    - Relative motion: subtract the start position so absolute location
      doesn't dominate the match.
    - Scale normalization: divide x/y by the max planar displacement so a 2 m
      throw and a 10 m throw share the same shape signature, while the
      *direction* (vertical drop vs horizontal drag) is preserved.
    - Speed profile channel: per-sample speed normalized by its mean. This
      separates flat-velocity motions (drag) from accelerating (drop) and
      spiky (rough handling, throw) motions even when their paths are similar.

    Note: estimated height (z_est) is deliberately NOT a matching channel in
    v1 because it is unavailable in the 2D warehouse footage; z feeds the
    physics stage instead.
    """
    if not points:
        return np.zeros((RESAMPLE_N, 3))
    pts = np.array([[p["x"], p["y"]] for p in points], dtype=float)
    if pts.shape[0] == 1:
        return np.zeros((RESAMPLE_N, 3))

    # Speed profile from positional finite differences (normalized by mean)
    ts = np.asarray([p.get("timestamp_sec", i * 0.2) for i, p in enumerate(points)], dtype=float)
    dt = np.diff(ts)
    dt = np.where(dt <= 0, 1e-6, dt)
    spd = np.linalg.norm(np.diff(pts, axis=0) / dt[:, None], axis=1)
    spd = np.concatenate([[spd[0]], spd])
    mean_spd = float(np.mean(spd)) + 1e-9
    spd_norm = spd / mean_spd

    # Relative + scale-normalized position shape
    rel = pts - pts[0]
    extent = float(np.max(np.linalg.norm(rel, axis=1))) + 1e-9
    rel = rel / extent

    feat = np.column_stack([rel[:, 0], rel[:, 1], spd_norm])
    n = feat.shape[0]
    if n >= RESAMPLE_N:
        idx = np.linspace(0, n - 1, RESAMPLE_N).astype(int)
        return feat[idx]
    old = np.linspace(0, n - 1, n)
    new = np.linspace(0, n - 1, RESAMPLE_N)
    return np.column_stack(
        [np.interp(new, old, feat[:, k]) for k in range(feat.shape[1])]
    )


def numpy_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Pure-numpy DTW distance (O(n*m)). Fine for short motion vectors."""
    n, m = a.shape[0], b.shape[0]
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    dtw = np.zeros((n + 1, m + 1), dtype=float)
    dtw[0, :] = np.inf
    dtw[:, 0] = np.inf
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(
                dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]
            )
    return float(dtw[n, m])


def dtw_distance(a: np.ndarray, b: np.ndarray, method: str = "fastdtw") -> float:
    """Compute DTW distance, preferring fastdtw when method == 'fastdtw'."""
    if method == "fastdtw":
        try:
            from fastdtw import fastdtw  # type: ignore

            a_list = a.tolist()
            b_list = b.tolist()
            dist, _path = fastdtw(a_list, b_list, dist=lambda x, y: float(np.hypot(x[0] - y[0], x[1] - y[1])))
            return float(dist)
        except ImportError:
            logger.debug("fastdtw not installed — using numpy DTW fallback.")
    return numpy_dtw(a, b)


class DTWMatcher:
    """Match one or many trajectories against the exemplar library."""

    def __init__(
        self,
        exemplar_path: str = "data/exemplars/behaviors.json",
        algorithm: str = "fastdtw",
        max_distance: float = 2.0,
        dtw_classes: Optional[List[str]] = None,
    ) -> None:
        self.library = ExemplarLibrary(exemplar_path)
        self.algorithm = algorithm
        self.max_distance = max_distance
        # Restrict DTW competition to the classes whose signature is a *motion
        # shape* (drop / drag / throw). Rule-driven behaviours (zone placement,
        # stacking, equipment, sequence) must not be matched by shape — a
        # straight-line exemplar would swallow every translation. When unset
        # (defaults, tests), all exemplars compete as before.
        self.dtw_classes = dtw_classes

    # ------------------------------------------------------------------ public

    def reload(self) -> None:
        self.library.reload()

    def match_trajectory(self, trajectory: Trajectory) -> BehaviorMatch:
        """Return the best exemplar match for a single trajectory."""
        vec = extract_trajectory_vector(
            [{"x": p.x, "y": p.y} for p in trajectory.points]
        )
        # Occasionally a zero-motion vector matches everything; guard:
        if float(np.hypot(vec[:, 0], vec[:, 1]).max()) < 1e-6:
            return BehaviorMatch(
                behavior_class="no_required_equipment",
                dtw_distance=self.max_distance,
                confidence=0.0,
                exemplar_id="",
                justification=(
                    "Trajectory has negligible displacement — no behaviour signal."
                ),
            )
        return self._best_match(vec, className=None)

    def match(self, trajectory: Trajectory, class_hint: Optional[str] = None) -> BehaviorMatch:
        """Alias of match_trajectory with an optional class filter."""
        vec = extract_trajectory_vector(
            [{"x": p.x, "y": p.y} for p in trajectory.points]
        )
        return self._best_match(vec, className=class_hint)

    def batch_match(self, trajectories: List[Trajectory]) -> List[BehaviorMatch]:
        return [self.match_trajectory(t) for t in trajectories]

    # ------------------------------------------------------------------ private

    def _best_match(
        self, vec: np.ndarray, className: Optional[str] = None
    ) -> BehaviorMatch:
        if not self.library.exemplars:
            return BehaviorMatch(
                behavior_class="unsafe_loading_sequence",
                dtw_distance=self.max_distance,
                confidence=0.0,
                exemplar_id="",
                justification="Exemplar library empty — cannot classify; "
                "flagged as generic unsafe loading.",
            )
        candidates = self.library.all()
        if self.dtw_classes:
            candidates = [
                e for e in candidates
                if e.get("behavior_class") in self.dtw_classes
            ]
        if className:
            candidates = [e for e in candidates if e.get("behavior_class") == className]

        best: Optional[BehaviorMatch] = None
        for exemplar in candidates:
            evec = extract_trajectory_vector(exemplar.get("points", []))
            dist = dtw_distance(vec, evec, self.algorithm)
            # Normalize by path length for comparability across exemplars
            norm = dist / RESAMPLE_N
            confidence = float(np.clip(1.0 / (1.0 + norm), 0.0, 1.0))
            candidate = BehaviorMatch(
                behavior_class=exemplar.get("behavior_class", "unsafe_loading_sequence"),
                dtw_distance=float(dist),
                confidence=confidence,
                exemplar_id=exemplar.get("id", ""),
                justification=(
                    f"Trajectory shape most closely matches exemplar "
                    f"'{exemplar.get('label', exemplar.get('id'))}' "
                    f"({exemplar.get('behavior_class')}) with DTW distance {dist:.3f} "
                    f"over {RESAMPLE_N} resampled points."
                ),
            )
            if best is None or candidate.dtw_distance < best.dtw_distance:
                best = candidate

        if best is None:
            best = BehaviorMatch(
                behavior_class="unsafe_loading_sequence",
                dtw_distance=self.max_distance,
                confidence=0.0,
                justification="No exemplars available to match against.",
            )
        # Gate on max_distance: too distant = not confident enough to alert
        if best.dtw_distance > self.max_distance * RESAMPLE_N:
            best.confidence = min(best.confidence, 0.2)
            best.justification += (
                " [WARNING: match distance exceeds configured threshold — "
                "low confidence]"
            )
        return best