"""RAG assistant over the ReplayTwin event database (Anthropic Tool Runner).

The assistant answers ONLY from structured event data. Its four tools query
the Postgres/SQLite events log; Claude can summarize, compare, and explain
those rows but never free-generates incident facts and never reads raw video.
Every tool return is compact JSON the model turns into a plain-English answer.

API surface (Anthropic Python SDK, official):
    - @beta_tool decorated functions define the tool schemas
    - client.beta.messages.tool_runner() drives the request->execute->loop
    - model = claude-opus-5

Degradations:
    - No ANTHROPIC_API_KEY     -> chat() returns a "not configured" reply.
    - Database unreachable     -> EventStore falls back to a local SQLite DB
      (see EventStore), so the assistant still answers from whatever events
      have been persisted.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import anthropic
    from anthropic import beta_tool
except Exception:  # pragma: no cover - SDK optional at runtime
    anthropic = None  # type: ignore
    beta_tool = None  # type: ignore

DEFAULT_MODEL = "claude-opus-5"

# ---------------------------------------------------------------------------
# Event store — the ONLY place tools may read facts from.
# ---------------------------------------------------------------------------

#: Queries use a conservative "days back" default if none passed in.
DEFAULT_RANGE_DAYS = 7


def _connect(db_url: str, sqlite_path: str):
    """Return a SQLAlchemy engine for the URL, with a SQLite fallback.

    Postgres keeps schema.sql semantics (UUID, JSONB); SQLite encodes those
    as TEXT so the demo runs without a live PostgreSQL server.
    """
    try:
        import sqlalchemy

        url = db_url or os.environ.get("EVENTS_DB_URL", "")
        if url:
            engine = sqlalchemy.create_engine(url)
            # touch the connection to confirm reachability
            with engine.connect():
                pass
            return engine, False
        raise RuntimeError("no database URL configured")
    except Exception as exc:
        logger.warning("Database %r unreachable (%s) — using SQLite %s.",
                       db_url or "(default)", exc, sqlite_path)
        import sqlalchemy

        return sqlalchemy.create_engine(f"sqlite:///{sqlite_path}"), True


class EventStore:
    """Structured queries over the events log (SQLAlchemy, PG or SQLite).

    Schema mirrors db/schema.sql (subset). Events stored with a string event
    id so SQLite (no native UUID) works unchanged.
    """

    def __init__(self, db_url: Optional[str] = None,
                 sqlite_path: str = "data/events.db") -> None:
        self.db_url = db_url
        self.sqlite_path = sqlite_path
        self.engine, self.is_sqlite = _connect(db_url, sqlite_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.engine.begin() as conn:
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    timestamp_sec REAL NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    source_video TEXT NOT NULL,
                    behavior_class TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    dtw_distance REAL,
                    physics_valid INTEGER NOT NULL DEFAULT 0,
                    physics_severity REAL,
                    justification TEXT NOT NULL DEFAULT '',
                    zone_id TEXT DEFAULT 'bay_0'
                )
                """
            )
            conn.exec_driver_sql(
                """
                CREATE INDEX IF NOT EXISTS idx_events_assistant_behavior
                ON events(behavior_class)
                """
            )
            conn.exec_driver_sql(
                """
                CREATE INDEX IF NOT EXISTS idx_events_assistant_created
                ON events(created_at)
                """
            )

    # ------------------------------------------------------------------ write

    def insert_event(self, event: Dict[str, Any]) -> None:
        """Persist a RiskEvent(-like dict). Used by the ingest backend."""
        row = {
            "id": event.get("event_id"),
            "event_id": event.get("event_id"),
            "created_at": event.get("created_at")
            or datetime.now(timezone.utc).isoformat(),
            "timestamp_sec": float(event.get("timestamp_sec", 0.0)),
            "frame_idx": int(event.get("frame_idx", 0)),
            "source_video": event.get("source_video", ""),
            "behavior_class": event.get("behavior_class", ""),
            "risk_score": float(event.get("risk_score", 0.0)),
            "risk_level": event.get("risk_level", "low"),
            "dtw_distance": event.get("dtw_distance"),
            "physics_valid": int(bool(event.get("physics_valid", True))),
            "physics_severity": event.get("physics_severity"),
            "justification": event.get("justification", ""),
            "zone_id": event.get("zone_id", "bay_0"),
        }
        cols = ", ".join(row.keys())
        marks = ", ".join("?" if self.is_sqlite else ":" + k for k in row)
        sql = (f"INSERT OR REPLACE INTO events ({cols}) VALUES ({marks})"
               if self.is_sqlite else
               f"INSERT INTO events ({cols}) VALUES ({marks}) "
               f"ON CONFLICT (id) DO UPDATE SET "
               + ", ".join(f"{k}=EXCLUDED.{k}" for k in row))
        with self.engine.begin() as conn:
            conn.exec_driver_sql(sql, [tuple(row[k] for k in row)])

    # ------------------------------------------------------------------ read

    _LEVEL_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def query_high_risk(self, days: int = DEFAULT_RANGE_DAYS,
                        min_level: str = "medium") -> List[dict]:
        """All events >= min_level within the last `days`."""
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=max(0, days))).isoformat()
        min_rank = self._LEVEL_RANK.get(min_level, 1)
        # Rank comparison in Python — the lexicographic TEXT order of the
        # levels ('critical' < 'medium') is the wrong order for a >= filter.
        sql = ("SELECT event_id, created_at, source_video, behavior_class, "
               "risk_score, risk_level, zone_id, justification "
               "FROM events WHERE created_at >= ? "
               "ORDER BY risk_score DESC LIMIT 500")
        with self.engine.connect() as conn:
            rows = conn.exec_driver_sql(sql, (cutoff,)).fetchall()
        out = [self._row(dict(zip(self._cols(rows), r))) for r in rows]
        return [e for e in out
                if self._LEVEL_RANK.get(e["risk_level"], 1) >= min_rank
                ][:200]

    def behavior_stats(self, behavior_class: str,
                       days: int = DEFAULT_RANGE_DAYS) -> List[dict]:
        """Aggregate counts and average risk for one behaviour class."""
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=max(0, days))).isoformat()
        sql = ("SELECT behavior_class, COUNT(*) AS n, "
               "AVG(risk_score) AS avg_risk, "
               "SUM(CASE WHEN risk_level IN ('high','critical') THEN 1 ELSE 0 END) "
               "AS severe_count "
               "FROM events WHERE behavior_class = ? AND created_at >= ? "
               "GROUP BY behavior_class")
        with self.engine.connect() as conn:
            rows = conn.exec_driver_sql(sql, (behavior_class, cutoff)).fetchall()
        return [self._row(dict(zip(self._cols(rows), r))) for r in rows]

    def bay_risk(self, bay_id: str, days: int = DEFAULT_RANGE_DAYS) -> List[dict]:
        """Events in a given bay/zone, newest first."""
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=max(0, days))).isoformat()
        sql = ("SELECT event_id, created_at, behavior_class, risk_score, "
               "risk_level, justification "
               "FROM events WHERE zone_id = ? AND created_at >= ? "
               "ORDER BY created_at DESC LIMIT 200")
        with self.engine.connect() as conn:
            rows = conn.exec_driver_sql(sql, (bay_id, cutoff)).fetchall()
        return [self._row(dict(zip(self._cols(rows), r))) for r in rows]

    def event_detail(self, event_id: str) -> Optional[dict]:
        """Full row for one event, or None."""
        sql = "SELECT * FROM events WHERE event_id = ? LIMIT 1"
        with self.engine.connect() as conn:
            rows = conn.exec_driver_sql(sql, (event_id,)).fetchall()
            if not rows:
                return None
            cols = list(rows[0]._mapping.keys())
        return self._row(dict(zip(cols, rows[0])))

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _row(d: dict) -> dict:
        """JSON-safe row (DAC values to floats/strings, strip unneeded)."""
        for k, v in list(d.items()):
            if v is None:
                continue
            if isinstance(v, (float, int)) and k in ("risk_score",):
                d[k] = round(float(v), 3)
        return d

    @staticmethod
    def _cols(rows: Any) -> List[str]:
        # Row._mapping: reliable column-key access across SQLAlchemy versions
        # (the cython Row rarely surfaces .keys()).
        return list(rows[0]._mapping.keys()) if rows else []


#: Module-level store the tools resolve against (set by the Assistant or ingest).
_STORE: Optional[EventStore] = None


def _dumps(rows: List[dict]) -> str:
    return json.dumps(rows, default=str)


# ---------------------------------------------------------------------------
# Tool definitions — structured Postgres/SQLite queries ONLY.
# ---------------------------------------------------------------------------

@beta_tool
def query_high_risk_events(time_range_days: int = 7, min_risk_level: str = "medium") -> str:
    """Query events at or above a risk level within a time range (days).

    Args:
        time_range_days: number of days back to look (e.g. 7 = last week).
        min_risk_level: one of "low", "medium", "high", "critical".
    """
    if _STORE is None:
        return "[]"
    try:
        return _dumps(_STORE.query_high_risk(time_range_days, min_risk_level))
    except Exception as exc:
        return f'{{"error": "query failed: {exc}"}}'


@beta_tool
def query_behavior_stats(behavior_class: str, time_range_days: int = 7) -> str:
    """Aggregate counts and average risk for one behavior class.

    Args:
        behavior_class: e.g. "product_dropped", "material_pushed_or_thrown",
            "product_dragged", "rough_handling".
        time_range_days: number of days back to look.
    """
    if _STORE is None:
        return "[]"
    try:
        return _dumps(_STORE.behavior_stats(behavior_class, time_range_days))
    except Exception as exc:
        return f'{{"error": "query failed: {exc}"}}'


@beta_tool
def query_bay_risk(bay_id: str, time_range_days: int = 7) -> str:
    """List events recorded for a specific bay/zone.

    Args:
        bay_id: zone identifier, e.g. "bay_1", "bay_7", "loading_dock".
        time_range_days: number of days back to look.
    """
    if _STORE is None:
        return "[]"
    try:
        return _dumps(_STORE.bay_risk(bay_id, time_range_days))
    except Exception as exc:
        return f'{{"error": "query failed: {exc}"}}'


@beta_tool
def explain_event(event_id: str) -> str:
    """Get the full stored record for a single event.

    Args:
        event_id: the event's id (shown in query results).
    """
    if _STORE is None:
        return "{}"
    try:
        detail = _STORE.event_detail(event_id)
        return _dumps([detail] if detail else [])
    except Exception as exc:
        return f'{{"error": "query failed: {exc}"}}'


TOOLS = [
    query_high_risk_events,
    query_behavior_stats,
    query_bay_risk,
    explain_event,
]

SYSTEM_PROMPT = (
    "You are the ReplayTwin warehouse safety assistant. Answer ONLY from the "
    "structured event data returned by your tools. You may summarize, count, "
    "compare, and explain those rows. Never invent incidents, counts, or "
    "severity levels that are not present in the tool results, and never "
    "reference video footage. If a tool returns no rows, say so plainly. "
    "Keep answers concise and cite the behaviour class and risk level you are "
    "referencing. Say which database fact supports each claim."
)


class Assistant:
    """Chat wrapper around the Anthropic Tool Runner.

    Args:
        model: Claude model id (default claude-opus-5).
        store: the EventStore the tools query. Created from config if None.
        api_key: explicit key; defaults to env ANTHROPIC_API_KEY.
        max_tokens: response cap for each model turn.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        store: Optional[EventStore] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 16000,
    ) -> None:
        global _STORE  # tools resolve against the module-level store
        self.model = model
        self.max_tokens = max_tokens
        self.store = store or _STORE or EventStore()
        _STORE = self.store
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None
        if anthropic is not None and self.api_key:
            self._client = anthropic.Anthropic(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(self, user_text: str) -> str:
        """Ask the assistant a question about warehouse events.

        Runs the tool runner to completion and returns the final answer text.
        """
        if not self.available:
            return (
                "Assistant is not configured: set ANTHROPIC_API_KEY in .env "
                "to enable the RAG assistant. The pipeline itself still works "
                "without it."
            )
        messages: List[dict] = [{"role": "user", "content": user_text}]
        try:
            runner = self._client.beta.messages.tool_runner(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            last = None
            for message in runner:
                last = message
            if last is None:
                return "Assistant produced no response."
            text = "".join(
                b.text for b in last.content if getattr(b, "type", "") == "text"
            ).strip()
            return text or "Assistant finished without a text answer."
        except Exception as exc:
            logger.warning("Assistant chat failed: %s", exc)
            return f"Assistant call failed: {exc}"