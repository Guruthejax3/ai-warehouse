"""Database engine/session for the backend.

Uses Postgres when DATABASE_URL is set (or the pipeline config), otherwise
falls back to a local SQLite file so the demo runs without a live server.
"""

from __future__ import annotations

import logging
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


def database_url() -> str:
    env = os.environ.get("DATABASE_URL", "").strip()
    if env:
        return env
    # Pipeline config may set a Postgres URL; only trust it if reachable,
    # else fall back to SQLite below.
    try:
        from pipeline.config import load_config

        cfg = load_config()
        return cfg["database"].get("url", "")
    except Exception:
        return ""


def make_engine():
    url = database_url()
    if url:
        try:
            engine = create_engine(url, pool_pre_ping=True)
            with engine.connect():
                pass
            logger.info("Using database %s", url.split("@")[-1])
            return engine
        except Exception as exc:
            logger.warning("Database %r unreachable (%s) — SQLite fallback.", url, exc)
    sqlite_path = os.environ.get("REPLAYTWIN_DB_SQLITE", "data/backend.db")
    os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
    return create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables (idempotent)."""
    from backend.models import Base

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()