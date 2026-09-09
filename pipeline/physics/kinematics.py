"""Physics analysis — velocity, acceleration, jerk, drop height, impacts.

This stage is the mandatory cross-check *before* any risk alert fires:
a behaviour match alone is never enough; matching kinematics must confirm it.

Calibration: pixel scale is derived from a known reference object
(e.g. a pallet that is 1.2 m wide and measured N pixels wide) configured in
config.yaml under physics.reference_object_*.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from pipeline.types import PhysicsState, Trajectory

logger = logging.getLogger(__name__)


class KinematicsAnalyzer:
    """Compute kinematic features and validate behaviour against physics.

    Args:
        pixel_scale: meters per pixel from reference-object calibration.
        gravity: m/s².
        drop_height_threshold: m — minimum estimated drop to call something
            "dropped".
        impact_velocity_threshold: m/s — minimum impact speed to accept an
            impact/drop behaviour.
        peak_velocity_threshold: m/s — unreasonable speeds are treated as
            sensing noise, not real impacts.
        jerk_threshold: m/s³ — minimum jerk to flag rough handling.
        mandatory_before_alert: if True (default), physics_valid is required
            for a risk event to be raised.
    """

    def __init__(
        self,
        pixel_scale: float = 0.01,
        gravity: float = 9.81,
        drop_height_threshold: float = 0.3,
        impact_velocity_threshold: float = 2.0,
        peak_velocity_threshold: float = 5.0,
        jerk_threshold: float = 10.0,
        mandatory_before_alert: bool = True,
    ) -> None:
        self.pixel_scale = pixel_scale
        self.gravity = gravity
        self.drop_height_threshold = drop_height_threshold
        self.impact_velocity_threshold = impact_velocity_threshold
        self.peak_velocity_threshold = peak_velocity_threshold
        self.jerk_threshold = jerk_threshold
        self.mandatory_before_alert = mandatory_before_alert

    # ------------------------------------------------------------------ public

    def analyze(self, trajectory: Trajectory) -> PhysicsState:
        """Compute physics for a trajectory and produce a PhysicsState."""
        n = len(trajectory.points)
        if n < 3:
            return PhysicsState(
                track_id=trajectory.track_id,
                physics_valid=False,
                severity_score=0.0,
                justification="Trajectory too short for physics analysis.",
            )

        ts = np.array([p.timestamp_sec for p in trajectory.points], dtype=float)
        xs = np.array([p.x for p in trajectory.points], dtype=float)
        ys = np.array([p.y for p in trajectory.points], dtype=float)
        zs = np.array([p.z_est for p in trajectory.points], dtype=float)

        # Finite-difference kinematics (gradient handles variable dt)
        vx = np.gradient(xs, ts)
        vy = np.gradient(ys, ts)
        # vertical component (depth/height) velocity
        vz = np.gradient(zs, ts) if np.any(zs) else np.zeros_like(zs)
        speed = np.hypot(vx, vy)

        ax = np.gradient(vx, ts)
        ay = np.gradient(vy, ts)
        accel = np.hypot(ax, ay)

        jx = np.gradient(ax, ts)
        jy = np.gradient(ay, ts)
        jerk = np.hypot(jx, jy)

        peak_velocity = float(np.max(speed))
        max_acceleration = float(np.max(accel))
        max_jerk = float(np.max(jerk))

        drop_height, drop_free_fall = self._estimate_drop_height(zs, vz, vy, speed)
        impact_speed = float(np.max(speed))

        # Generic dynamical validity: the trajectory must describe real,
        # plausible physical motion (moving, and not absurdly fast).
        physics_valid = bool(
            peak_velocity > 0.05
            and peak_velocity <= self.peak_velocity_threshold
            and max_acceleration <= 20 * self.gravity
        )

        severity = self._severity(
            peak_velocity, max_acceleration, max_jerk,
            drop_height, impact_speed,
        )

        features = {
            "peak_velocity": peak_velocity,
            "max_acceleration": max_acceleration,
            "max_jerk": max_jerk,
            "drop_height": drop_height,
            "free_fall": drop_free_fall,
            "impact_speed": impact_speed,
            "speed_profile_std": float(np.std(speed)),
        }

        justification = (
            f"Physics measurements: peak speed {peak_velocity:.2f} m/s, "
            f"max acceleration {max_acceleration:.1f} m/s², "
            f"max jerk {max_jerk:.1f} m/s³, estimated drop height "
            f"{drop_height:.2f} m (free-fall-consistent: {drop_free_fall}), "
            f"impact speed {impact_speed:.1f} m/s. "
            f"Motion is {'physically plausible' if physics_valid else 'NOT physically plausible (sensor noise or no motion)'}."
        )

        return PhysicsState(
            track_id=trajectory.track_id,
            peak_velocity=peak_velocity,
            max_acceleration=max_acceleration,
            max_jerk=max_jerk,
            drop_height_est=drop_height,
            collision_detected=bool(impact_speed >= self.impact_velocity_threshold),
            collision_severity=min(1.0, impact_speed / (3 * self.gravity)),
            physics_valid=physics_valid,
            severity_score=severity,
            justification=justification,
            features=features,
        )

    def cross_check(
        self, behavior_class: str, physics: PhysicsState
    ) -> tuple:
        """Class-specific validation: does physics support this behaviour?

        Returns (ok: bool, reasons: List[str]). This is the mandatory gate
        the risk scorer calls before raising an alert.

        Signatures:
            product_dropped / material_pushed_or_thrown  -> needs drop/impact
            product_dragged                              -> sustained, plausible speed
            rough_handling                               -> high jerk
            *stacking / *zone / *equipment               -> spatial, need motion
            unsafe_loading_sequence                      -> motion + height change
        """
        f = physics.features
        ok = True
        reasons: List[str] = []

        if not physics.physics_valid:
            ok = False
            reasons.append("trajectory not physically plausible")

        # Every class requires a minimal amount of motion
        min_motion = 0.05
        if f.get("peak_velocity", 0.0) < min_motion:
            ok = False
            reasons.append(f"peak speed {f.get('peak_velocity', 0):.2f} m/s below "
                           f"minimum motion of {min_motion} m/s")

        if behavior_class in (
            "product_dropped", "material_pushed_or_thrown",
        ):
            drop_ok = f.get("drop_height", 0.0) >= self.drop_height_threshold
            impact_ok = (
                self.impact_velocity_threshold
                <= f.get("impact_speed", 0.0)
                <= self.peak_velocity_threshold
            )
            if not drop_ok:
                ok = False
                reasons.append(f"drop height {f.get('drop_height', 0):.2f} m below "
                               f"threshold {self.drop_height_threshold} m "
                               f"for {behavior_class}")
            if not impact_ok:
                ok = False
                reasons.append(f"impact speed {f.get('impact_speed', 0):.1f} m/s not "
                               f"in [{self.impact_velocity_threshold}, "
                               f"{self.peak_velocity_threshold}] m/s for {behavior_class}")

        elif behavior_class == "product_dragged":
            if f.get("peak_velocity", 0.0) > 0.8 * self.peak_velocity_threshold:
                ok = False
                reasons.append(f"speed {f.get('peak_velocity', 0):.2f} m/s too high "
                               f"for a drag (expected slow sustained motion)")

        elif behavior_class == "rough_handling":
            if f.get("max_jerk", 0.0) < self.jerk_threshold:
                ok = False
                reasons.append(f"max jerk {f.get('max_jerk', 0):.1f} m/s³ below "
                               f"rough-handling threshold {self.jerk_threshold} m/s³")

        elif behavior_class == "unsafe_loading_sequence":
            height_change = f.get("drop_height", 0.0)
            if height_change < self.drop_height_threshold:
                ok = False
                reasons.append(f"no significant height change "
                               f"({height_change:.2f} m) to confirm a loading "
                               f"sequence violation")

        if physics.physics_valid and ok:
            reasons.append("physics confirms the behaviour signature")
        return ok, reasons

    # ------------------------------------------------------------------ internals

    def _estimate_drop_height(
        self, zs: np.ndarray, vz: np.ndarray, vy: np.ndarray, speed: np.ndarray
    ) -> tuple:
        """Estimate drop height from vertical motion + free-fall consistency.

        Two cases:
        - Depth channel present (zs non-zero): height from the z range and
          free-fall velocity,	gated on actual vertical displacement.
        - No depth channel (the common 2D warehouse camera): a fall is
          inferred from a downward burst in the screen-y velocity (image y
          grows downward), and the height comes from the free-fall relation
          h = v²/2g using the impact speed. Purely horizontal motion has
          no downward burst and correctly scores no drop.
        """
        if not np.any(zs):
            # 2D footage: downward y-velocity marks the fall direction.
            max_down = float(np.max(np.maximum(vy, 0))) if np.any(vy > 0) else 0.0
            peak_speed = float(np.max(speed))
            # Fall detected when the downward burst is clearly above sensor
            # noise (half the impact threshold).
            if max_down >= 0.5 * self.impact_velocity_threshold:
                h = (max(max_down, peak_speed) ** 2) / (2 * self.gravity)
                return min(h, 3.0), True
            return 0.0, False

        z_range = float(np.max(zs) - np.min(zs))
        descending = float(np.max(np.abs(np.minimum(vz, 0)))) if np.any(vz < 0) else 0.0
        free_fall = descending**2 / (2 * self.gravity) + 1e-6
        if free_fall > self.drop_height_threshold + 0.05:
            return min(z_range if z_range > 0 else free_fall, 3.0), True
        return float(max(0.0, z_range)), False

    def _severity(
        self, peak_v, max_a, max_j, drop_h, impact_speed
    ) -> float:
        """0-1 physics severity (used by risk scorer as w2 component)."""
        v_s = min(1.0, peak_v / (2 * self.impact_velocity_threshold))
        a_s = min(1.0, max_a / (4 * self.gravity))
        d_s = min(1.0, drop_h / 1.5)
        j_s = min(1.0, max_j / (10 * self.jerk_threshold))
        i_s = min(1.0, impact_speed / (2 * self.impact_velocity_threshold))
        return float(np.clip(0.25 * v_s + 0.2 * a_s + 0.2 * d_s + 0.15 * j_s + 0.2 * i_s, 0.0, 1.0))