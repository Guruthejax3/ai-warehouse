"""SQLAlchemy ORM models mirroring db/schema.sql.

Tables: events, trajectories, physics_states, exemplars, feedback.

Layout notes:
    - ``events.id`` is a UUID surrogate PK (gen_random_uuid on Postgres).
    - ``events.event_id`` is the pipeline's human-facing id (RiskEvent.event_id)
      and is UNIQUE — this is what the assistant/API expose.
    - JSON columns use sqlalchemy.JSON so the models work on both Postgres
      and the SQLite fallback.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def gen_uuid() -> uuid.UUID:
    return uuid.uuid4()


class Event(Base):
    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=gen_uuid)
    event_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    timestamp_sec: Mapped[float] = mapped_column(Float)
    frame_idx: Mapped[int] = mapped_column(Integer)
    source_video: Mapped[str] = mapped_column(Text)
    behavior_class: Mapped[str] = mapped_column(String(64), index=True)
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)
    event_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    motion_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dtw_distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    physics_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    physics_severity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    justification: Mapped[str] = mapped_column(Text, default="")
    evidence_clip_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    evidence_clip_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    evidence_clip_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    zone_id: Mapped[str] = mapped_column(String(32), default="bay_0", index=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    trajectories: Mapped[List["TrajectoryModel"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )
    physics_states: Mapped[List["PhysicsStateModel"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )


class TrajectoryModel(Base):
    __tablename__ = "trajectories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=gen_uuid)
    event_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), nullable=True
    )
    track_id: Mapped[int] = mapped_column(Integer)
    class_label: Mapped[str] = mapped_column(String(32), default="object")
    points: Mapped[List[dict]] = mapped_column(JSON, default=list)
    total_displacement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    event: Mapped[Optional["Event"]] = relationship(back_populates="trajectories")


class PhysicsStateModel(Base):
    __tablename__ = "physics_states"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=gen_uuid)
    event_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), nullable=True
    )
    track_id: Mapped[int] = mapped_column(Integer)
    peak_velocity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_acceleration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_jerk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    drop_height_est: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    collision_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    collision_severity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    severity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    justification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    event: Mapped[Optional["Event"]] = relationship(back_populates="physics_states")


class Exemplar(Base):
    __tablename__ = "exemplars"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    behavior_class: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(Text)
    trajectory_data: Mapped[List[dict]] = mapped_column(JSON, default=list)
    source_clip: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint("event_id", "annotator", name="uq_feedback_event_annotator"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=gen_uuid)
    event_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), nullable=True
    )
    annotator: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    correct_behavior: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    correct_risk_level: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )