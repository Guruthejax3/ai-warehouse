#!/usr/bin/env python3
"""Seed the exemplar library with synthetic-but-plausible trajectories.

Each of the 10 behaviour classes gets a kinematically sensible exemplar:
motion shapes are derived from physical signatures (drag = constant low
velocity, throw = ballistic arc, drop = accelerating descent + impact, etc.).

    usage: python scripts/seed_exemplars.py [output.json]

Run once to (re)generate data/exemplars/behaviors.json. Real exemplars can
later be added via scripts/tag_exemplar.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DEFAULT_OUT = ROOT / "data" / "exemplars" / "behaviors.json"

FPS = 5.0
GRAVITY = 9.81


def _pts(xs, ys, zs=None, vxs=None, vys=None):
    """Pack arrays into the exemplar points schema."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = xs.shape[0]
    zs = np.zeros(n) if zs is None else np.asarray(zs, dtype=float)
    vxs = np.gradient(xs, 1.0 / FPS) if vxs is None else np.asarray(vxs, dtype=float)
    vys = np.gradient(ys, 1.0 / FPS) if vys is None else np.asarray(vys, dtype=float)
    return [
        {
            "frame_idx": int(i),
            "timestamp_sec": round(i / FPS, 3),
            "x": round(float(xs[i]), 4),
            "y": round(float(ys[i]), 4),
            "z_est": round(float(zs[i]), 4),
            "vx": round(float(vxs[i]), 4),
            "vy": round(float(vys[i]), 4),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Behaviour generators
# ---------------------------------------------------------------------------

def product_dragged():
    """Sustained constant low-speed lateral translation (cupboard drag)."""
    n = 40
    t = np.arange(n)
    xs = t * 0.08
    ys = np.sin(t * 0.03) * 0.4
    return _pts(xs, ys), "Product pulled along floor at steady low speed with slight lateral wobble", {"fragility": "standard"}


def product_dropped():
    """Accelerating vertical descent (near free fall) then abrupt stop."""
    n = 16
    t = np.arange(n) / FPS
    h0 = 1.2
    z = h0 - 0.5 * GRAVITY * t**2
    z = np.clip(z, 0, h0)
    xs = np.ones(n) * 3.0
    ys = 4.0 + 0.5 * GRAVITY * t**2  # accelerating fall in the y (down-frame) axis
    return _pts(xs, ys, zs=z), "Object falls vertically, accelerating, then stops abruptly on impact (estimated drop 1.2 m)", {"fragility": "fragile"}


def rough_handling():
    """High jerk, erratic direction reversals (slamming down repeatedly)."""
    n = 30
    t = np.arange(n)
    xs = 2.0 + np.sin(t * 0.5) * 0.8
    ys = 3.0 + np.sin(t * 1.3) * 1.2
    return _pts(xs, ys), "Erratic jerky motion with repeated sharp direction changes and high acceleration peaks", {"fragility": "fragile"}


def incorrect_stacking():
    """Heavy box placed on top of a lighter-looking lower stack (damage zone)."""
    n = 22
    t = np.arange(n)
    xs = np.linspace(2.0, 2.2, n) + np.sin(t) * 0.02
    ys = 2.0 + np.maximum(t * 0.12, 0)  # lifts then settles on a stack
    z = np.zeros(n)
    z[8:] = 0.6  # rests noticeably higher = stacked on something
    return _pts(xs, ys, zs=z), "Object lifted and set down onto an existing stack — heavy-on-light hazard in stacking zone", {"fragility": "standard"}


def unstable_stacking():
    """Wobble: small high-frequency oscillation on top of slow drift."""
    n = 40
    t = np.arange(n)
    xs = 5.0 + t * 0.01 + np.sin(t * 1.2) * 0.15
    ys = 3.0 + np.sin(t * 0.2) * 0.2 + np.sin(t * 2.0) * 0.08
    return _pts(xs, ys), "Object oscillates increasingly while resting — unstable, top-heavy stack", {"fragility": "very_fragile"}


def outside_designated_zone():
    """Leaves a bounded region: crosses zone boundary and continues."""
    n = 50
    t = np.arange(n)
    xs = 1.0 + t * 0.12
    ys = 4.0 + np.sin(t * 0.1) * 0.15
    return _pts(xs, ys), "Object travels past the designated storage boundary line and continues outside the zone", {"fragility": "standard"}


def no_required_equipment():
    """Slow carry with trunk-bend signature (no forklift/cart in path)."""
    n = 45
    t = np.arange(n)
    xs = t * 0.05
    ys = 1.5 + np.sin(t * 0.05) * 0.6  # up-down bend while carrying
    return _pts(xs, ys), "Slow manual carry with repeated bending (no mechanical equipment in the motion path)", {"fragility": "standard"}


def pallet_mispositioned():
    """Long slow translation, then stop at the wrong offset (misaligned pallet)."""
    n = 55
    t = np.arange(n)
    xs = np.minimum(t * 0.1, 4.4)
    ys = 2.5 + np.sin(t * 0.08) * 0.3
    return _pts(xs, ys), "Pallet travels and stops misaligned relative to the expected dock mark, angled from the guide line", {"fragility": "standard"}


def material_pushed_or_thrown():
    """Ballistic launch: fast initial velocity that decays over distance."""
    n = 18
    t = np.arange(n) / FPS
    v0 = 3.0
    decel = 1.2
    xs = v0 * t - 0.5 * decel * t**2  # released fast, then slows (friction/gravity)
    xs = np.clip(xs, 0, None)
    ys = 3.0 - 0.15 * t  # slight downward drift in flight
    z = np.clip(1.0 - 0.5 * GRAVITY * t**2, 0, 1.0)
    return _pts(xs, ys, zs=z), "Object launched with a fast ballistic burst that decays as it travels — pushed or thrown, not carried", {"fragility": "fragile"}


def unsafe_loading_sequence():
    """Rapid lift-drop-lift that crashes onto unstable base within a short window."""
    n = 26
    t = np.arange(n)
    xs = 2.0 + np.minimum(t * 0.05, 0.8)
    ys = 1.0 + np.abs(np.sin(t * 0.6)) * 0.6
    z = 0.0
    z = np.where((t >= 8) & (t < 12), 0.9, z)   # high lift
    z = np.where((t >= 12) & (t < 16), 0.0, z)  # crash drop
    z = np.where(t >= 16, 0.4, z)               # partial recovery onto lower base
    return _pts(xs, ys, zs=z), "Rapid lift → forceful drop → partial re-load onto a lower, unstable base within seconds", {"fragility": "fragile"}


GENERATORS = [
    product_dragged,
    product_dropped,
    rough_handling,
    incorrect_stacking,
    unstable_stacking,
    outside_designated_zone,
    no_required_equipment,
    pallet_mispositioned,
    material_pushed_or_thrown,
    unsafe_loading_sequence,
]

BEHAVIOR_CLASS_TO_GEN = {
    "product_dragged": product_dragged,
    "product_dropped": product_dropped,
    "rough_handling": rough_handling,
    "incorrect_stacking": incorrect_stacking,
    "unstable_stacking": unstable_stacking,
    "outside_designated_zone": outside_designated_zone,
    "no_required_equipment": no_required_equipment,
    "pallet_mispositioned": pallet_mispositioned,
    "material_pushed_or_thrown": material_pushed_or_thrown,
    "unsafe_loading_sequence": unsafe_loading_sequence,
}


def seed_all(output: Path) -> None:
    """Write one exemplar per behaviour class to the library file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"exemplars": [], "version": "1.0"}
    for generator in GENERATORS:
        points, label, metadata = generator()
        cls = generator.__name__
        payload["exemplars"].append(
            {
                "id": f"{cls}_seed_001",
                "behavior_class": cls,
                "label": label,
                "points": points,
                "source_clip": None,
                "metadata": metadata,
            }
        )
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Seeded {len(payload['exemplars'])} exemplars -> {output}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    seed_all(out)