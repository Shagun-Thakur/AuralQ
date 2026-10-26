"""Unit tests for SQLite Session Persistence Engine (Milestone E7)."""

import tempfile
from pathlib import Path
import pytest

from src.agent.db import SessionDatabase
from src.agent.schemas import CitationMatch, EvidencePackage, GroundingReport, TurnRecord
from src.llm.schemas import IntentParseResult, ToolCallIntent


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = SessionDatabase(db_path=db_path, enabled=True)
    yield db
    try:
        Path(db_path).unlink(missing_ok=True)
    except PermissionError:
        pass


def test_session_db_initialization(temp_db):
    """Verify database schema creation and empty state."""
    sessions = temp_db.list_sessions()
    assert sessions == []


def test_save_and_retrieve_turn(temp_db):
    """Verify storing and deserializing a complete conversational turn."""
    intent = IntentParseResult(
        user_intent="measure brightness",
        reasoning="spectral centroid tool is best fit",
        selected_tools=[ToolCallIntent(tool_name="spectral_centroid", parameters={})],
        is_abstention=False,
    )
    evidence = EvidencePackage(
        query="Does this sound bright?",
        status="SUCCESS",
        tools_called=["spectral_centroid"],
        consolidated_metrics={"spectral_centroid.mean_hz": 4200.5},
        consolidated_anchors={"spectral_centroid.perceived_brightness": "bright"},
        all_citation_aliases={"spectral_centroid.mean_hz": ["4200.5 Hz", "4.2 kHz"]},
        total_dsp_latency_ms=3.5,
    )
    grounding = GroundingReport(
        is_fully_grounded=True,
        grounding_score=1.0,
        total_numbers_cited=1,
        verified_citations=[
            CitationMatch(
                raw_token="4200.5 Hz",
                parsed_value=4200.5,
                matched_metric="spectral_centroid.mean_hz",
                match_type="exact_alias",
                is_grounded=True,
            )
        ],
        hallucinated_numbers=[],
        was_repaired=False,
    )

    turn = TurnRecord(
        turn_id=1,
        timestamp=1700000000.0,
        query="Does this sound bright?",
        parsed_intent=intent,
        evidence=evidence,
        response_text="The audio sounds bright with a spectral centroid of 4200.5 Hz.",
        latency_breakdown={"intent_ms": 2500.0, "dsp_ms": 3.5, "synthesis_ms": 2800.0, "total_ms": 5303.5},
        cached_tools_used=[],
        grounding_report=grounding,
    )

    session_id = "sess_001"
    temp_db.create_or_update_session(
        session_id=session_id,
        title="Brightness Analysis",
        audio_path="test_audio.wav",
        is_dataset=False,
    )
    temp_db.save_turn(session_id, turn)

    # Verify session list
    sessions = temp_db.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
    assert sessions[0]["title"] == "Brightness Analysis"
    assert sessions[0]["audio_path"] == "test_audio.wav"
    assert sessions[0]["turn_count"] == 1

    # Verify turn deserialization
    turns = temp_db.get_session_turns(session_id)
    assert len(turns) == 1
    loaded = turns[0]
    assert loaded.turn_id == 1
    assert loaded.query == "Does this sound bright?"
    assert loaded.response_text == "The audio sounds bright with a spectral centroid of 4200.5 Hz."
    assert loaded.parsed_intent.selected_tools[0].tool_name == "spectral_centroid"
    assert loaded.evidence.consolidated_metrics["spectral_centroid.mean_hz"] == 4200.5
    assert loaded.grounding_report.is_fully_grounded is True


def test_session_deletion_cascade(temp_db):
    """Verify that deleting a session cascades and removes all associated turns."""
    session_id = "sess_to_delete"
    temp_db.create_or_update_session(session_id, "Temporary Session")

    turn = TurnRecord(
        turn_id=1,
        timestamp=1700000000.0,
        query="Test query",
        evidence=EvidencePackage(query="Test query", status="SUCCESS"),
        response_text="Test response",
        latency_breakdown={"total_ms": 10.0},
    )
    temp_db.save_turn(session_id, turn)
    assert len(temp_db.get_session_turns(session_id)) == 1

    # Delete session
    temp_db.delete_session(session_id)
    assert temp_db.list_sessions() == []
    assert temp_db.get_session_turns(session_id) == []


def test_ephemeral_cloud_mode():
    """Verify that enabled=False acts as a pure in-memory no-op (Zero Cloud Data Retention)."""
    db = SessionDatabase(enabled=False)
    assert db.enabled is False
    assert db.db_path is None
    assert db.list_sessions() == []
    assert db.get_session_turns("any_id") == []
    assert db.save_turn("any_id", None) is True
    assert db.delete_session("any_id") is True


def test_cloud_environment_hard_disable(monkeypatch):
    """Verify that cloud environment detection forcibly overrides enabled=True to False."""
    monkeypatch.setenv("AURALQ_FORCE_CLOUD", "1")
    db = SessionDatabase(enabled=True)
    assert db.enabled is False
    assert db.db_path is None
    assert db.list_sessions() == []
    assert db.get_session_turns("any_id") == []
    assert db.save_turn("any_id", None) is True
    assert db.create_or_update_session("any_id", "Title") is True
    assert db.delete_session("any_id") is True
