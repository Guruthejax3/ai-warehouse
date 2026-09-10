"""Tests for pipeline.rules.zones — the deterministic zone-placement rule.

Zone polygons are normalised 0-1 coordinates. The trajectory stores meters,
which ZoneRuleChecker converts to pixels→normalised for the polygon test.
"""

import math

import pytest

from pipeline.rules.zones import ZoneRuleChecker, _FULL_FRAME
from pipeline.types import Trajectory, TrajectoryPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _traj(cx_m: float, cy_m: float, n: int = 20, pixel_scale: float = 0.01) -> Trajectory:
    """Build a minimal trajectory whose final centroid is (cx_m, cy_m) meters."""
    pts = []
    for i in range(n):
        t = i / max(n - 1, 1)
        pts.append(
            TrajectoryPoint(
                frame_idx=i,
                timestamp_sec=t * 2.0,
                x=cx_m,
                y=cy_m,
                z_est=0.0,
                vx=0.0,
                vy=0.0,
            )
        )
    return Trajectory(track_id="zone_test", points=pts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestZoneRuleChecker:
    """Zone rule fires when final centroid is outside every configured polygon,
    and stays dormant when zones are the full-frame default."""

    def test_default_full_frame_polygon_dormant(self):
        """With only _FULL_FRAME polygon, no event fires — rule is dormant."""
        checker = ZoneRuleChecker({"default": [_FULL_FRAME]})
        traj = _traj(5.0, 3.5)  # 500 px, 350 px → normalised (0.5, 0.5) — inside
        result = checker.check(traj, 1280, 720, 0.01)
        assert result is None

    def test_outside_small_polygon_fires(self):
        """Final centroid outside a small central polygon triggers event."""
        small_poly = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]
        checker = ZoneRuleChecker({"default": [small_poly]})
        # (9.0, 6.0) m → 900 px, 600 px → norm (0.703, 0.833) → outside
        traj = _traj(9.0, 6.0)
        result = checker.check(traj, 1280, 720, 0.01)
        assert result is not None
        assert result["behavior_class"] == "outside_designated_zone"
        assert result["severity"] > 0.0
        assert "Rule fired" not in result["reason"]  # reason is raw, not justification

    def test_inside_small_polygon_no_event(self):
        """Final centroid inside the polygon → no event."""
        small_poly = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
        checker = ZoneRuleChecker({"default": [small_poly]})
        # (6.0, 4.0) m → 600 px, 400 px → norm (0.469, 0.556) → inside
        traj = _traj(6.0, 4.0)
        result = checker.check(traj, 1280, 720, 0.01)
        assert result is None

    def test_no_zones_configured_dormant(self):
        """Empty zone config → rule dormant (no event)."""
        checker = ZoneRuleChecker({})
        traj = _traj(9.0, 6.0)
        result = checker.check(traj, 1280, 720, 0.01)
        assert result is None

    def test_severity_increases_with_distance(self):
        """A point farther outside should have higher severity."""
        small_poly = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
        checker = ZoneRuleChecker({"default": [small_poly]})

        # Slightly outside
        close = _traj(7.5, 5.0)  # norm (0.586, 0.694) — just outside
        r1 = checker.check(close, 1280, 720, 0.01)

        # Far outside
        far = _traj(11.0, 6.5)  # norm (0.859, 0.903) — far outside
        r2 = checker.check(far, 1280, 720, 0.01)

        assert r1 is not None and r2 is not None
        assert r2["severity"] >= r1["severity"]

    def test_multiple_polygons_inside_any_passes(self):
        """Two polygons — point inside the second → no event."""
        poly_a = [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]
        poly_b = [[0.7, 0.7], [0.9, 0.7], [0.9, 0.9], [0.7, 0.9]]
        checker = ZoneRuleChecker({"default": [poly_a, poly_b]})
        # (9.0, 6.0) m → 900 px, 600 px → norm (0.703, 0.833) → inside poly_b
        traj = _traj(9.0, 6.0)
        result = checker.check(traj, 1280, 720, 0.01)
        assert result is None

    def test_multiple_polygons_outside_all_fires(self):
        """Two polygons — point outside both → event fires."""
        poly_a = [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]
        poly_b = [[0.8, 0.8], [1.0, 0.8], [1.0, 1.0], [0.8, 1.0]]
        checker = ZoneRuleChecker({"default": [poly_a, poly_b]})
        # (5.0, 4.0) m → 500 px, 400 px → norm (0.391, 0.556) → outside both
        traj = _traj(5.0, 4.0)
        result = checker.check(traj, 1280, 720, 0.01)
        assert result is not None
        assert result["behavior_class"] == "outside_designated_zone"

    def test_severity_bounded_zeroToOne(self):
        """Severity always stays in [0, 1]."""
        tiny_poly = [[0.49, 0.49], [0.51, 0.49], [0.51, 0.51], [0.49, 0.51]]
        checker = ZoneRuleChecker({"default": [tiny_poly]})
        # Way outside
        traj = _traj(12.0, 7.0)
        result = checker.check(traj, 1280, 720, 0.01)
        assert result is not None
        assert 0.0 <= result["severity"] <= 1.0
