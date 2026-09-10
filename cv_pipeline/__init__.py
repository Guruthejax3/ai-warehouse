"""cv-pipeline — six standalone, runnable CV stages.

Each stage is a dependency-light module with its own ``run_<stage>.py`` CLI
and a JSON handoff contract, so a reviewer can run every stage on its own:

    1. motion_detection   -> motion regions + event_type (sustained/spike)
    2. detection_tracking -> persistent object tracks (IoU + Kalman)
    3. pose_estimation    -> body keypoints + hazard posture flags
    4. kinematics         -> velocity/acceleration/jerk/drop-height/collision
    5. behaviour_fsm      -> activity FSM, unsafe loading-sequence detection
    6. risk_scoring       -> weighted, explainable risk score per event

The ``run_pipeline`` orchestrator chains all six over one video for an
end-to-end report. The integrated ``pipeline/`` package (the backend's
entry point) reuses these primitives; this directory is the reviewer-facing
stage-by-stage implementation.
"""

from cv_pipeline.motion_detection.motion_detector import MotionDetector, MotionRegion, MotionResult
from cv_pipeline.detection_tracking.detection_tracking import Detector, Tracker, Track
from cv_pipeline.pose_estimation.pose_estimation import PoseEstimator
from cv_pipeline.kinematics.kinematics import KinematicsAnalyzer
from cv_pipeline.behaviour_fsm.behaviour_fsm import BehaviourFSM
from cv_pipeline.risk_scoring.risk_scoring import RiskScorer

__all__ = [
    "MotionDetector", "MotionRegion", "MotionResult",
    "Detector", "Tracker", "Track",
    "PoseEstimator",
    "KinematicsAnalyzer",
    "BehaviourFSM",
    "RiskScorer",
]