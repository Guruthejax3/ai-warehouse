"""Shared data types for the ReplayTwin pipeline.

All pipeline modules communicate through these typed dataclasses.
This ensures consistent interfaces between perception, trajectory,
matching, physics, and risk stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Perception types
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """A single tracked object detected in one frame."""
    track_id: int
    bbox: List[int]               # [x1, y1, x2, y2]
    class_label: str              # "person" | "object" | "forklift" | "pallet"
    area: int                     # bbox area in pixels
    confidence: float = 1.0       # detection confidence (0-1)
    centroid: Optional[Tuple[float, float]] = None  # (cx, cy)


@dataclass
class Keypoint:
    """A single body keypoint with coordinates and visibility."""
    x: float                      # normalized 0-1
    y: float                      # normalized 0-1
    z: float                      # estimated depth
    visibility: float             # confidence 0-1


@dataclass
class PoseKeypoints:
    """Pose keypoints for a single person."""
    track_id: int
    keypoints: List[Keypoint]     # 33 MediaPipe landmarks
    bbox: List[int]               # [x1, y1, x2, y2]
    angles: Dict[str, float] = field(default_factory=dict)  # computed joint angles
    source: str = "mediapipe"     # "mediapipe" | "opencv_fallback"


# ---------------------------------------------------------------------------
# Trajectory types
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    """A single point in a trajectory."""
    frame_idx: int
    timestamp_sec: float
    x: float                      # position x (pixels or meters)
    y: float                      # position y (pixels or meters)
    z_est: float = 0.0            # estimated depth/height
    vx: float = 0.0              # velocity x
    vy: float = 0.0              # velocity vy


@dataclass
class Trajectory:
    """Complete trajectory for one tracked object."""
    track_id: int
    points: List[TrajectoryPoint] = field(default_factory=list)
    class_label: str = "object"
    total_displacement: float = 0.0
    mean_speed: float = 0.0
    max_speed: float = 0.0

    @property
    def length(self) -> int:
        return len(self.points)


# ---------------------------------------------------------------------------
# Matching types
# ---------------------------------------------------------------------------

BEHAVIOR_CLASSES = [
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
]


@dataclass
class BehaviorMatch:
    """Result of matching a trajectory against exemplars."""
    behavior_class: str           # one of BEHAVIOR_CLASSES
    dtw_distance: float           # raw DTW distance (lower = better match)
    confidence: float             # 0-1, derived from distance
    exemplar_id: str = ""
    justification: str = ""       # human-readable explanation


# ---------------------------------------------------------------------------
# Physics types
# ---------------------------------------------------------------------------

@dataclass
class PhysicsState:
    """Physics analysis results for a trajectory."""
    track_id: int
    peak_velocity: float = 0.0    # m/s
    max_acceleration: float = 0.0 # m/s²
    max_jerk: float = 0.0         # m/s³
    drop_height_est: float = 0.0  # meters
    collision_detected: bool = False
    collision_severity: float = 0.0
    physics_valid: bool = True    # mandatory gate — must be True to alert
    severity_score: float = 0.0   # 0-1, overall physics severity
    justification: str = ""
    features: Dict[str, float] = field(default_factory=dict)  # raw kinematics


# ---------------------------------------------------------------------------
# Risk types
# ---------------------------------------------------------------------------

RiskLevel = Literal["low", "medium", "high", "critical"]


@dataclass
class RiskEvent:
    """A scored risk event — the final output of the pipeline."""
    event_id: str
    timestamp_sec: float
    frame_idx: int
    source_video: str
    behavior_class: str
    risk_score: float             # 0-1
    risk_level: RiskLevel
    dtw_match: Optional[BehaviorMatch] = None
    physics: Optional[PhysicsState] = None
    justification: str = ""       # MUST be non-empty — explainable alert
    evidence_clip_start: int = 0  # frame index
    evidence_clip_end: int = 0    # frame index
    trajectory_id: int = 0
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline frame-level result (handoff between stages)
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Per-frame result carrying data through the pipeline."""
    frame_idx: int
    timestamp_sec: float
    # Motion detection (from MotionDetector)
    motion_regions: List[dict] = field(default_factory=list)
    frame_motion_score: float = 0.0
    motion_event_type: str = "none"
    # Perception
    tracked_objects: List[TrackedObject] = field(default_factory=list)
    poses: Dict[int, PoseKeypoints] = field(default_factory=dict)
    # Full pipeline result (populated at end)
    risk_events: List[RiskEvent] = field(default_factory=list)
