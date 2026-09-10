"""Pallet stability assessment — score load stability from trajectory physics.

A pallet's stability is estimated from the trajectory dynamics of the
loading process. Key signals:

    - vertical jitter during placement (high jitter = unstable)
    - lateral drift (sliding = shifting load)
    - speed consistency (variable speed = impact risk)
    - stack height estimate (from z_est trajectory range)

The output is a stability score (0 = unstable, 1 = perfectly stable) with
a structured justification that the dashboard can display.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StabilityAssessment:
    """Result of pallet stability analysis."""
    stability_score: float        # 0.0 (unstable) to 1.0 (stable)
    tier: str                     # unstable | marginal | stable | very_stable
    jitter_score: float           # vertical movement consistency
    drift_score: float            # lateral sliding risk
    speed_consistency: float      # speed variance (lower = better)
    height_estimate: float        # estimated stack height from trajectory z
    justification: str
    factors: dict = field(default_factory=dict)


def assess_pallet_stability(
    trajectory_points: List[dict],
    pixel_scale: float = 0.01,
    gravity: float = 9.81,
) -> StabilityAssessment:
    """Assess pallet loading stability from a trajectory's physics data.

    Args:
        trajectory_points: list of dicts with keys x, y, z_est (optional),
            timestamp_sec, and optionally speed, acceleration.
        pixel_scale: metres per pixel for unit conversion.
        gravity: gravity constant for normalisation.

    Returns:
        StabilityAssessment with scores and justification.
    """
    if len(trajectory_points) < 3:
        return StabilityAssessment(
            stability_score=0.5, tier="marginal",
            jitter_score=0.5, drift_score=0.5, speed_consistency=0.5,
            height_estimate=0.0,
            justification="Insufficient trajectory data for stability assessment (need >= 3 points).",
        )

    # Extract coordinates
    xs = np.array([p.get("x", 0.0) for p in trajectory_points], dtype=float)
    ys = np.array([p.get("y", 0.0) for p in trajectory_points], dtype=float)
    ts = np.array([p.get("timestamp_sec", 0.0) for p in trajectory_points], dtype=float)

    # Z-estimate (vertical position in 3D, or fallback to inverted y)
    zs = np.array([p.get("z_est", 0.0) for p in trajectory_points], dtype=float)
    if np.all(zs == 0):
        zs = -ys  # fallback: higher y = lower in image = higher in world

    # --- Jitter: vertical acceleration variance ---
    dt = np.diff(ts)
    dt = np.maximum(dt, 1e-6)
    vz = np.diff(zs) / dt
    az = np.diff(vz) / np.maximum(dt[1:], 1e-6)
    jitter_rms = float(np.sqrt(np.mean(az ** 2))) if len(az) > 0 else 0.0
    # Normalise: 0 m/s^2 = perfect, >5 m/s^2 = very jittery
    jitter_score = max(0.0, min(1.0, 1.0 - jitter_rms / (gravity * 0.5)))

    # --- Drift: lateral (x) displacement variance ---
    vx = np.diff(xs) / dt
    drift_rms = float(np.sqrt(np.mean(vx ** 2))) * pixel_scale
    # Normalise: 0 m/s = no drift, >0.3 m/s = significant
    drift_score = max(0.0, min(1.0, 1.0 - drift_rms / 0.3))

    # --- Speed consistency: coefficient of variation of speed ---
    ds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) * pixel_scale
    speeds = ds / dt
    mean_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    std_speed = float(np.std(speeds)) if len(speeds) > 0 else 0.0
    cv = std_speed / max(mean_speed, 1e-6)
    speed_consistency = max(0.0, min(1.0, 1.0 - cv))

    # --- Height estimate: range of z trajectory ---
    height_px = float(np.max(zs) - np.min(zs)) * pixel_scale

    # --- Composite stability score ---
    stability = (
        0.35 * jitter_score
        + 0.25 * drift_score
        + 0.30 * speed_consistency
        + 0.10 * max(0.0, 1.0 - height_px / 2.0)  # shorter stacks more stable
    )
    stability = max(0.0, min(1.0, stability))

    if stability < 0.3:
        tier = "unstable"
    elif stability < 0.6:
        tier = "marginal"
    elif stability < 0.85:
        tier = "stable"
    else:
        tier = "very_stable"

    justification = (
        f"Pallet stability: {tier} ({stability:.0%}). "
        f"Vertical jitter score {jitter_score:.2f}, lateral drift score "
        f"{drift_score:.2f}, speed consistency {speed_consistency:.2f}. "
        f"Estimated stack height: {height_px:.2f}m."
    )
    if tier == "unstable":
        justification += " HIGH RISK: secure the load before transport."
    elif tier == "marginal":
        justification += " Monitor: consider adding bands or re-stacking."

    return StabilityAssessment(
        stability_score=round(stability, 4),
        tier=tier,
        jitter_score=round(jitter_score, 4),
        drift_score=round(drift_score, 4),
        speed_consistency=round(speed_consistency, 4),
        height_estimate=round(height_px, 4),
        justification=justification,
        factors={
            "jitter_rms_ms2": round(jitter_rms, 4),
            "drift_rms_ms": round(drift_rms, 4),
            "speed_cv": round(cv, 4),
            "height_m": round(height_px, 4),
        },
    )
