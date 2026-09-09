"""Unit tests for the physics/kinematics stage — regression coverage for the
2D drop-height estimation (a real warehouse clip where a mattress was thrown
was being gated: z_est is always 0 in 2D footage, so the fall was invisible to
the old vz-only estimator)."""

from __future__ import annotations

import numpy as np

from pipeline.physics.kinematics import KinematicsAnalyzer
from pipeline.types import Trajectory, TrajectoryPoint

GRAVITY = 9.81


def _traj(xs, ys, fps=5.0):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    return Trajectory(
        track_id=1,
        points=[
            TrajectoryPoint(
                frame_idx=i,
                timestamp_sec=i / fps,
                x=float(xs[i]),
                y=float(ys[i]),
            )
            for i in range(len(xs))
        ],
    )


def _accelerating_drop(n=16, v_impact=2.6):
    """Fast downward burst then a sudden stop (the recognizable 2D impact):
    at 5 fps a real free-fall from ~0.35 m is over in under two frames, so the
    tracker sees a few frames at v_impact (image y grows down) then a halt.
    z_est stays zero — this is the 'Throwing Mattresses' signature."""
    ys = np.full(n, 100.0)
    for k in range(1, 4):
        ys[5 + k:] += v_impact / 5.0  # 3 consecutive downward bursts of Δy
    return _traj(np.full(n, 100.0), ys)


def _horizontal_drag(n=40):
    """Slow pure-horizontal drag: no downward burst, must NOT register a drop."""
    t = np.arange(n)
    xs = t * 0.08
    ys = np.full(n, 50.0)
    return _traj(xs, ys, fps=5.0)


def test_2d_drop_estimates_height_without_depth_channel():
    """A fall along screen-y with peak speed ~2.2 m/s yields a drop height
    consistent with free fall (v²/2g ≈ 0.25 m) and physics_valid True."""
    analyzer = KinematicsAnalyzer()
    traj = _accelerating_drop()
    ph = analyzer.analyze(traj)
    # estimated height ≈ v²/2g = 0.345 m for v_impact=2.6, within reason
    assert 0.25 <= ph.drop_height_est <= 0.45
    assert ph.physics_valid


def test_2d_drop_passes_product_dropped_cross_check():
    """Regression for the gated-throw bug: impact at ~2.2 m/s over a measurable
    fall must NOT be blocked by the physics gate for product_dropped."""
    analyzer = KinematicsAnalyzer(impact_velocity_threshold=2.0)
    traj = _accelerating_drop()
    ph = analyzer.analyze(traj)
    ok, reasons = analyzer.cross_check("product_dropped", ph)
    assert ok, f"expected drop to pass physics gate, got: {reasons}"


def test_2d_horizontal_drag_is_not_a_drop():
    """A pure horizontal drag has no downward burst → no drop height → the
    drop/push classes stay gated (no false alerts on normal carrying)."""
    analyzer = KinematicsAnalyzer()
    traj = _horizontal_drag()
    ph = analyzer.analyze(traj)
    assert ph.drop_height_est == 0.0
    ok, _ = analyzer.cross_check("product_dropped", ph)
    assert not ok