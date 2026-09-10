"""ReplayTwin — Warehouse Video Intelligence Pipeline.

Pipeline stages:
  1. Motion Detection    → cv_pipeline/motion_detection/
  2. Segmentation        → pipeline/perception/segmentation.py
  3. Pose Estimation     → pipeline/perception/pose.py
  4. Trajectory Building → pipeline/trajectory/builder.py
  5. Behaviour Matching  → pipeline/matching/dtw_matcher.py
  6. Physics Analysis    → pipeline/physics/kinematics.py
  7. Risk Scoring        → pipeline/risk/scoring.py
  8. Replay Generation   → pipeline/replay/generator.py
  9. Voice Coaching      → pipeline/voice/coach.py
  10. RAG Assistant      → pipeline/assistant/rag.py
"""

__version__ = "0.1.0"
