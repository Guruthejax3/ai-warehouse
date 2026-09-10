"""Unit tests for behaviour matching (DTW) against the exemplar library."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.matching.dtw_matcher import (
    DTWMatcher,
    extract_trajectory_vector,
    numpy_dtw,
)
from pipeline.types import Trajectory, TrajectoryPoint

FPS = 5.0


def _traj(xs, ys):
    pts = [
        TrajectoryPoint(
            frame_idx=i, timestamp_sec=i / FPS, x=float(xs[i]), y=float(ys[i])
        )
        for i in range(len(xs))
    ]
    return Trajectory(track_id=1, points=pts)


def _ballistic_throw(n=18):
    t = np.arange(n) / FPS
    xs = 3.0 * t - 0.5 * 1.2 * t**2
    ys = 3.0 - 0.15 * t
    return _traj(xs, ys)


def _steady_drag(n=40):
    t = np.arange(n) / FPS
    return _traj(0.08 * n * t * FPS * 0.2, np.sin(t * FPS * 0.03) * 0.4)


def _vertical_drop(n=16):
    t = np.arange(n) / FPS
    return _traj(np.ones(n) * 3.0, 4.0 + 0.5 * 9.81 * t**2)


# ---------------------------------------------------------------------------
# Vector extraction
# ---------------------------------------------------------------------------

def test_vector_shape_and_normalization():
    vec = extract_trajectory_vector(
        [{"x": 1.0 + i, "y": 2.0, "timestamp_sec": i * 0.2} for i in range(40)]
    )
    # Fixed-length, 3 channels (x, y, speed)
    assert vec.shape == (32, 3)
    # Scale-normalized: max planar displacement ≈ 1
    assert np.hypot(vec[:, 0], vec[:, 1]).max() == pytest.approx(1.0, abs=0.05)


def test_vector_direction_preserved():
    """A horizontal move and a vertical move must not collapse to the same shape."""
    horiz = extract_trajectory_vector(
        [{"x": i, "y": 1.0, "timestamp_sec": i * 0.2} for i in range(40)]
    )
    vert = extract_trajectory_vector(
        [{"x": 1.0, "y": i, "timestamp_sec": i * 0.2} for i in range(40)]
    )
    # Horizontal: x dominates; Vertical: y dominates
    assert horiz[:, 0].max() > 0.8
    assert vert[:, 1].max() > 0.8
    assert numpy_dtw(horiz, vert) > 1.0


def test_zero_motion_guard():
    vec = extract_trajectory_vector([{"x": 1.0, "y": 1.0} for _ in range(20)])
    assert vec.shape == (32, 3)
    assert np.all(np.isfinite(vec))


# ---------------------------------------------------------------------------
# Matching behaviour
# ---------------------------------------------------------------------------

def test_throw_matches_pushed_or_thrown():
    matcher = DTWMatcher()
    result = matcher.match_trajectory(_ballistic_throw())
    assert result.behavior_class == "material_pushed_or_thrown"


def test_drop_matches_product_dropped():
    matcher = DTWMatcher()
    result = matcher.match_trajectory(_vertical_drop())
    assert result.behavior_class == "product_dropped"


def test_throw_and_drag_discriminate():
    """Speed profile must separate a fast decaying throw from a steady drag."""
    matcher = DTWMatcher()
    assert matcher.match_trajectory(_ballistic_throw()).behavior_class == (
        "material_pushed_or_thrown"
    )
    # A steady drag close to the drag exemplar should NOT be flagged as a throw
    drag = matcher.match_trajectory(_steady_drag())
    assert drag.behavior_class != "material_pushed_or_thrown"


def test_confidence_bounded():
    matcher = DTWMatcher()
    result = matcher.match_trajectory(_ballistic_throw())
    assert 0.0 <= result.confidence <= 1.0
    assert result.justification  # always explainable


def test_empty_library_falls_back():
    matcher = DTWMatcher()
    matcher.library.exemplars = []  # simulate empty library
    result = matcher.match_trajectory(_ballistic_throw())
    assert result.behavior_class == "unsafe_loading_sequence"
    assert result.justification  # explainable even in fallback


def test_batch_match():
    matcher = DTWMatcher()
    results = matcher.batch_match([_ballistic_throw(), _vertical_drop()])
    assert [r.behavior_class for r in results] == [
        "material_pushed_or_thrown",
        "product_dropped",
    ]
