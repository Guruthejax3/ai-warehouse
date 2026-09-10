"""Pydantic request/response schemas for the ReplayTwin backend."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class IngestRequest(BaseModel):
    video_path: str
    max_frames: int = 0  # 0 = whole clip


class IngestResponse(BaseModel):
    video_path: str
    total: int
    events_persisted: List[str]


class AssistantRequest(BaseModel):
    message: str


class AssistantResponse(BaseModel):
    message: str


class VoiceRequest(BaseModel):
    behavior_class: str
    risk_level: str = "medium"
    physics_gated: bool = False
    speak: bool = False  # response text only by default; TTS is behind a flag
    language: str = "en"  # multilingual alerts: en | hi | te | es


class VoiceResponse(BaseModel):
    message: str
    spoken: bool


class FeedbackRequest(BaseModel):
    event_id: Optional[str] = None
    annotator: Optional[str] = None
    correct_behavior: Optional[str] = None
    correct_risk_level: Optional[str] = None
    notes: Optional[str] = None