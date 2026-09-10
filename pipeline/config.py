"""Config loader — reads config.yaml into a typed dict with sensible defaults.

All pipeline modules pull their settings from here rather than hardcoding.
Missing keys degrade gracefully to defaults so the pipeline never crashes
on bootstrap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "pipeline": {"target_fps": 5.0, "video_source": "data/clips/"},
    "motion_detection": {
        "method": "farneback",
        "motion_threshold": 2.0,
        "min_region_area": 500,
        "spike_std_multiplier": 2.5,
        "rolling_window": 15,
    },
    "segmentation": {
        "enabled": True,
        "use_sam2": False,
        "fallback": "opencv_fallback",
        "merge_kernel_px": 6,
        "max_new_tracks": 12,
    },
    "pose_estimation": {"enabled": True, "use_mediapipe": True, "fallback": "none"},
    "trajectory": {
        "smoother": "moving_average",
        "ma_window": 5,
        "min_track_length": 10,
        "min_speed_mps": 0.05,
        "max_gap_frames": 5,
    },
    "matching": {
        "algorithm": "fastdtw",
        "exemplar_path": "data/exemplars/behaviors.json",
        "max_distance": 2.0,
        "behavior_classes": [
            "product_dropped",
            "product_dragged",
            "rough_handling",
            "incorrect_stacking",
            "unstable_stacking",
            "outside_designated_zone",
            "no_required_equipment",
            "pallet_mispositioned",
            "material_pushed_or_thrown",
            "unsafe_loading_sequence",
        ],
    },
    "physics": {
        "mandatory_before_alert": True,
        "gravity": 9.81,
        "pixel_scale": 0.01,
        "reference_object_width_px": 120,
        "reference_object_width_m": 1.2,
        "drop_height_threshold": 0.3,
        "impact_velocity_threshold": 2.0,
        "peak_velocity_threshold": 5.0,
        "jerk_threshold": 10.0,
    },
    "risk": {
        "weights": {
            "w1_dtw_match": 0.25,
            "w2_physics_severity": 0.25,
            "w3_fragility": 0.15,
            "w4_repetition": 0.20,
            "w5_zone_criticality": 0.15,
        },
        "levels": {"low": 0.25, "medium": 0.50, "high": 0.75, "critical": 1.00},
        "evidence_trim_sec": 5,
        "fragility_classes": {"standard": 0.3, "fragile": 0.7, "very_fragile": 1.0},
    },
    "voice": {"enabled": False, "engine": "pyttsx3", "rate": 150},
    "storage": {"use_s3": False, "s3_bucket": "damage-dna-evidence", "s3_endpoint": None},
    "database": {"url": "postgresql://localhost:5432/damage_dna"},
    "assistant": {"model": "claude-opus-5", "enabled": False},
    "server": {"host": "0.0.0.0", "port": 8000, "ws_heartbeat_sec": 30},
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config.yaml merged over built-in defaults.

    Args:
        path: optional explicit config path. Defaults to ./config.yaml.

    Returns:
        A merged config dict (never None).
    """
    cfg = _deep_merge({}, DEFAULTS)  # deep copy of defaults
    config_path = path or Path(__file__).parent.parent / "config.yaml"
    if not Path(config_path).exists():
        logger.warning("config.yaml not found at %s — using defaults.", config_path)
        return cfg
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        _deep_merge(cfg, loaded)
    except Exception as exc:  # never crash on a broken config
        logger.warning("Failed to load config.yaml (%s) — using defaults.", exc)
    return cfg