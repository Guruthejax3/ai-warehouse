"""Pipeline orchestrator — one clip through the full chain, returning data.

This is the shared entry point used by the CLI (scripts/run_pipeline.py) and
the FastAPI backend (POST /api/ingest). It consumes the existing upstream
MotionDetector from cv_pipeline/motion_detection via sys.path.

    events, trajectories = process_clip(video_path, cfg)

Events are fully-scored RiskEvent objects with justifications; trajectories
are the sealed Trajectory objects (used for replay payloads).
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT / "cv_pipeline" / "motion_detection") not in sys.path:
    sys.path.insert(0, str(ROOT / "cv_pipeline" / "motion_detection"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_detector import MotionDetector  # noqa: E402

from pipeline.activity.classifier import ActivityClassifier  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.matching.dtw_matcher import DTWMatcher  # noqa: E402
from pipeline.perception.pose import PoseEstimator  # noqa: E402
from pipeline.perception.segmentation import SegmentationTracker  # noqa: E402
from pipeline.physics.kinematics import KinematicsAnalyzer  # noqa: E402
from pipeline.risk.scoring import RiskScorer  # noqa: E402
from pipeline.rules.zones import ZoneRuleChecker  # noqa: E402
from pipeline.sequences.detector import SequenceDetector  # noqa: E402
from pipeline.trajectory.builder import TrajectoryBuilder  # noqa: E402
from pipeline.types import (
    BehaviorMatch,
    PhysicsState,
    RiskEvent,
    Trajectory,
)  # noqa: E402

logger = logging.getLogger(__name__)


def _build_stages(cfg: Dict):
    """Instantiate all pipeline stages from config."""
    fps = float(cfg["pipeline"]["target_fps"])
    motion = MotionDetector(method="farneback")
    seg = SegmentationTracker(
        target_fps=fps,
        merge_kernel_px=int(cfg["segmentation"].get("merge_kernel_px", 6)),
        max_new_tracks=int(cfg["segmentation"].get("max_new_tracks", 12)),
    )
    builder = TrajectoryBuilder(
        smoother=cfg["trajectory"].get("smoother", "moving_average"),
        ma_window=int(cfg["trajectory"].get("ma_window", 5)),
        min_track_length=int(cfg["trajectory"].get("min_track_length", 10)),
        min_speed_mps=float(cfg["trajectory"].get("min_speed_mps", 0.05)),
        pixel_scale=float(cfg["physics"].get("pixel_scale", 0.01)),
        max_gap=int(cfg["trajectory"].get("max_gap_frames", 10)),
    )
    matcher = DTWMatcher(
        exemplar_path=cfg["matching"].get("exemplar_path", "data/exemplars/behaviors.json"),
        algorithm=cfg["matching"].get("algorithm", "fastdtw"),
        dtw_classes=list(cfg["matching"].get("dtw_classes") or []),
    )
    zone_rule = ZoneRuleChecker(cfg.get("zones", {}))
    physics = KinematicsAnalyzer(
        pixel_scale=float(cfg["physics"].get("pixel_scale", 0.01)),
        gravity=float(cfg["physics"].get("gravity", 9.81)),
    )
    risk = RiskScorer(
        weights=cfg["risk"].get("weights"),
        fragility_classes=cfg["risk"].get("fragility_classes"),
        mandatory_before_alert=bool(cfg["physics"].get("mandatory_before_alert", True)),
        evidence_trim_sec=int(cfg["risk"].get("evidence_trim_sec", 5)),
        target_fps=fps,
    )
    activity = ActivityClassifier(
        pixel_scale=float(cfg["physics"].get("pixel_scale", 0.01)),
        dock_edge=cfg.get("activity", {}).get("dock_edge", "auto"),
    )
    sequence = SequenceDetector(
        target_fps=fps,
        truck_edge=cfg.get("activity", {}).get("dock_edge", "auto"),
        enabled=bool(cfg.get("sequence", {}).get("enabled", True)),
        pixel_scale=float(cfg["physics"].get("pixel_scale", 0.01)),
    )
    return motion, seg, builder, matcher, physics, risk, zone_rule, activity, sequence


def process_clip(
    video_path: Path,
    cfg: Optional[Dict] = None,
    max_frames: int = 0,
    on_event: Optional[Callable[[RiskEvent], None]] = None,
) -> Tuple[List[RiskEvent], List[Trajectory]]:
    """Run the whole pipeline on one clip.

    Args:
        video_path: path to a warehouse clip.
        cfg: optional config dict (loads ./config.yaml if None).
        max_frames: stop after N processed frames (0 = whole clip).
        on_event: optional callback invoked per scored event.

    Returns:
        (events, trajectories) — events sorted by risk descending.
    """
    cfg = cfg or load_config()
    fps = float(cfg["pipeline"]["target_fps"])
    pixel_scale = float(cfg["physics"].get("pixel_scale", 0.01))
    (
        motion, seg, builder, matcher, physics, risk, zone_rule, activity,
        sequence,
    ) = _build_stages(cfg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    prev = None
    frame_idx = 0  # processed-frame count (drives the synthetic clock)
    src_idx = 0    # source-frame count (for sampling every `step`)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        src_idx += 1
        if (src_idx - 1) % step != 0:
            continue
        if max_frames and frame_idx >= max_frames:
            break
        frame_idx += 1
        # Synthetic clock on the *processed* timeline: the source is sampled
        # every `step` frames, so ts = processed_count / target_fps. Using the
        # source-frame counter inflated timestamps by `step` (40 frames looked
        # like 47 s into a ~9 s clip), which broke evidence-clip trimming.
        ts = frame_idx / fps
        if prev is None:
            prev = frame
            continue
        res = motion.process_frame_pair(prev, frame)
        tracked = seg.process(frame, res.motion_regions, frame_idx, ts)
        builder.update(tracked, frame_idx, ts)
        prev = frame
    cap.release()

    trajectories = builder.finalize()
    events: List[RiskEvent] = []

    # Sequence-of-actions detection: an unsafe loading chain (approach ->
    # handle -> transit -> place) fires a dedicated sequence event. This runs
    # over *all* sealed trajectories (multi-frame, multi-object evidence), not
    # a single trajectory, satisfying the "sequence of actions" requirement.
    hypotheses = sequence.detect(trajectories, frame_w, frame_h)
    for hyp in hypotheses:
        if not hyp.matched:
            continue
        last_ts = hyp.steps[-1]["end_sec"] if hyp.steps else 0.0
        # Score like any other rule-driven event (physics already verified at
        # the trajectory level; a full chain is inherently high-severity).
        scored = risk.score_event(
            source_video=str(video_path),
            frame_idx=int(last_ts * fps),
            timestamp_sec=last_ts,
            behavior=BehaviorMatch(
                behavior_class="unsafe_loading_sequence",
                dtw_distance=0.0,
                confidence=hyp.confidence,
                exemplar_id="",
                justification=hyp.justification,
            ),
            physics=PhysicsState(track_id=-1, physics_valid=True,
                                 severity_score=0.7),
            physics_ok=True,
            fragility="standard",
            zone_criticality=0.35,
            rule_reason="sequence FSM fired: unsafe loading sequence "
                        f"(steps observed: {[s['step'] for s in hyp.steps]})",
        )
        scored.metadata.update({
            "sequence": hyp.hypothesis,
            "steps": hyp.steps,
            "order_respected": hyp.order_respected,
            "sequence_confidence": round(hyp.confidence, 3),
        })
        events.append(scored)
        if on_event:
            on_event(scored)
    for traj in trajectories:
        behavior = matcher.match_trajectory(traj)
        ph = physics.analyze(traj)
        ok, reasons = physics.cross_check(behavior.behavior_class, ph)
        act_type, act_conf, act_just = activity.classify(traj)
        ev = risk.score_event(
            source_video=str(video_path),
            frame_idx=int(traj.points[-1].frame_idx),
            timestamp_sec=float(traj.points[-1].timestamp_sec),
            behavior=behavior,
            physics=ph,
            physics_ok=ok,
            physics_reasons=reasons,
        )
        ev.activity_type = act_type
        ev.metadata["activity_type"] = act_type
        ev.metadata["activity_confidence"] = round(act_conf, 3)
        ev.metadata["activity_justification"] = act_just
        events.append(ev)
        if on_event:
            on_event(ev)

        # Deterministic layout rules (zone placement) fire independently of DTW.
        zone_trigger = zone_rule.check(traj, frame_w, frame_h, pixel_scale)
        if zone_trigger:
            rule_ev = risk.score_event(
                source_video=str(video_path),
                frame_idx=int(traj.points[-1].frame_idx),
                timestamp_sec=float(traj.points[-1].timestamp_sec),
                behavior=BehaviorMatch(
                    behavior_class=zone_trigger["behavior_class"],
                    dtw_distance=0.0,
                    confidence=1.0,
                    exemplar_id="",
                    justification=zone_trigger["reason"],
                ),
                physics=ph,
                physics_ok=True,
                fragility="standard",
                zone_criticality=zone_trigger["severity"],
                rule_reason=zone_trigger["reason"],
            )
            rule_ev.activity_type = act_type
            rule_ev.metadata["activity_type"] = act_type
            rule_ev.metadata["activity_justification"] = act_just
            events.append(rule_ev)
            if on_event:
                on_event(rule_ev)

    events.sort(key=lambda e: e.risk_score, reverse=True)
    return events, trajectories