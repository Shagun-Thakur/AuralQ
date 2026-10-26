"""SQLite Session Persistence Engine for AuralQ.

Manages persistent local storage of conversational turns, tool executions, and
grounding reports for private local sessions. Provides an ephemeral/disabled
mode for Cloud Demo deployments to guarantee zero server-side data retention.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.agent.schemas import TurnRecord
from src.agent.state import SessionState


def is_cloud_environment() -> bool:
    """Detect if running in a public Cloud deployment (e.g. Streamlit Community Cloud or AURALQ_FORCE_CLOUD)."""
    return Path("/mount/src").exists() or os.environ.get("AURALQ_FORCE_CLOUD", "").lower() in ("1", "true")


class SessionDatabase:
    """
    SQLite-backed session manager for local private AuralQ workspaces.
    When `enabled=False` (e.g. in Cloud Demo Mode), operates purely as an
    in-memory/no-op storage to prevent any server-side data retention.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        enabled: bool = True,
    ):
        # Strict guarantee: Zero database retention in cloud deployments
        if is_cloud_environment():
            enabled = False

        self.enabled = enabled
        if self.enabled:
            self.db_path = Path(db_path or "data/auralq_sessions.db")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
        else:
            self.db_path = None

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create, configure, and safely close a SQLite connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema if it does not already exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    audio_path TEXT,
                    is_dataset INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    intent_json TEXT,
                    evidence_json TEXT,
                    grounding_json TEXT,
                    latency_json TEXT,
                    cached_tools_json TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
            """)

            # Backfill legacy sessions where audio_path was not recorded
            try:
                null_rows = conn.execute(
                    "SELECT session_id FROM sessions WHERE audio_path IS NULL"
                ).fetchall()
                for r in null_rows:
                    sid = r["session_id"]
                    t_row = conn.execute(
                        "SELECT evidence_json FROM turns WHERE session_id = ? AND evidence_json IS NOT NULL ORDER BY turn_id DESC LIMIT 1",
                        (sid,),
                    ).fetchone()
                    if t_row and t_row["evidence_json"]:
                        ev = json.loads(t_row["evidence_json"])
                        recovered_path = (ev.get("audio_metadata") or {}).get("file_path") or ev.get("audio_id")
                        if recovered_path:
                            conn.execute(
                                "UPDATE sessions SET audio_path = ? WHERE session_id = ?",
                                (recovered_path, sid),
                            )
            except Exception:
                pass

    def create_or_update_session(
        self,
        session_id: str,
        title: str,
        audio_path: Optional[str] = None,
        is_dataset: bool = False,
    ) -> bool:
        """Create a new session entry or update an existing session's metadata."""
        if not self.enabled:
            return True

        import time
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, created_at, audio_path, is_dataset)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title = CASE WHEN sessions.title IS NOT NULL AND sessions.title != '' THEN sessions.title ELSE excluded.title END,
                    audio_path = COALESCE(excluded.audio_path, sessions.audio_path),
                    is_dataset = excluded.is_dataset;
                """,
                (session_id, title, time.time(), audio_path, 1 if is_dataset else 0),
            )
        return True

    def save_turn(
        self,
        session_id: str,
        turn: TurnRecord,
        audio_path: Optional[str] = None,
        is_dataset: bool = False,
    ) -> bool:
        """Persist a completed conversational turn into SQLite."""
        if not self.enabled or turn is None:
            return True

        # If audio_path is not explicitly provided, attempt extraction from turn evidence
        if audio_path is None and turn.evidence:
            if turn.evidence.audio_metadata and turn.evidence.audio_metadata.file_path:
                audio_path = turn.evidence.audio_metadata.file_path
            elif turn.evidence.audio_id and Path(turn.evidence.audio_id).exists():
                audio_path = turn.evidence.audio_id

        # Ensure parent session exists and has its audio record linked
        title = turn.query[:40] + ("..." if len(turn.query) > 40 else "")
        self.create_or_update_session(
            session_id=session_id,
            title=title,
            audio_path=audio_path,
            is_dataset=is_dataset,
        )

        intent_json = json.dumps(turn.parsed_intent.model_dump()) if turn.parsed_intent else None
        evidence_json = json.dumps(turn.evidence.model_dump()) if turn.evidence else None
        grounding_json = json.dumps(turn.grounding_report.model_dump()) if turn.grounding_report else None
        latency_json = json.dumps(turn.latency_breakdown)
        cached_tools_json = json.dumps(turn.cached_tools_used)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO turns (
                    turn_id, session_id, query, response_text,
                    intent_json, evidence_json, grounding_json,
                    latency_json, cached_tools_json, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    turn.turn_id,
                    session_id,
                    turn.query,
                    turn.response_text,
                    intent_json,
                    evidence_json,
                    grounding_json,
                    latency_json,
                    cached_tools_json,
                    turn.timestamp,
                ),
            )
        return True

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List past sessions ordered by creation date descending."""
        if not self.enabled:
            return []

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.audio_path, s.is_dataset,
                       COUNT(t.id) as turn_count
                FROM sessions s
                LEFT JOIN turns t ON s.session_id = t.session_id
                GROUP BY s.session_id
                ORDER BY s.created_at DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "session_id": r["session_id"],
                    "title": r["title"],
                    "created_at": r["created_at"],
                    "audio_path": r["audio_path"],
                    "is_dataset": bool(r["is_dataset"]),
                    "turn_count": r["turn_count"],
                }
                for r in rows
            ]

    def get_session_turns(self, session_id: str) -> List[TurnRecord]:
        """Fetch all turn records for a specific session."""
        if not self.enabled:
            return []

        from src.agent.schemas import EvidencePackage, GroundingReport
        from src.llm.schemas import IntentParseResult

        turns: List[TurnRecord] = []
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT turn_id, query, response_text, intent_json,
                       evidence_json, grounding_json, latency_json,
                       cached_tools_json, timestamp
                FROM turns
                WHERE session_id = ?
                ORDER BY turn_id ASC;
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            for r in rows:
                intent = (
                    IntentParseResult.model_validate(json.loads(r["intent_json"]))
                    if r["intent_json"]
                    else None
                )
                evidence = (
                    EvidencePackage.model_validate(json.loads(r["evidence_json"]))
                    if r["evidence_json"]
                    else None
                )
                grounding = (
                    GroundingReport.model_validate(json.loads(r["grounding_json"]))
                    if r["grounding_json"]
                    else None
                )
                latency = json.loads(r["latency_json"]) if r["latency_json"] else {}
                cached_tools = json.loads(r["cached_tools_json"]) if r["cached_tools_json"] else []

                turn = TurnRecord(
                    turn_id=r["turn_id"],
                    timestamp=r["timestamp"],
                    query=r["query"],
                    parsed_intent=intent,
                    evidence=evidence,
                    response_text=r["response_text"],
                    latency_breakdown=latency,
                    cached_tools_used=cached_tools,
                    grounding_report=grounding,
                )
                turns.append(turn)
        return turns

    def load_session(self, session_id: str) -> Optional[Tuple[Dict[str, Any], List[TurnRecord]]]:
        """Load session metadata and all associated turns."""
        if not self.enabled:
            return None

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT session_id, title, created_at, audio_path, is_dataset FROM sessions WHERE session_id = ?;",
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            session_meta = {
                "session_id": row["session_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "audio_path": row["audio_path"],
                "is_dataset": bool(row["is_dataset"]),
            }

        turns = self.get_session_turns(session_id)
        return session_meta, turns

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its associated turns."""
        if not self.enabled:
            return True

        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
        return True

    def clear_all_sessions(self) -> bool:
        """Clear all stored sessions and turns (for local reset)."""
        if not self.enabled:
            return True

        with self._get_connection() as conn:
            conn.execute("DELETE FROM turns;")
            conn.execute("DELETE FROM sessions;")
        return True
