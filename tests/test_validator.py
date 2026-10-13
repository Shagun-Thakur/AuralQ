"""Unit and integration test suite for Milestone E5: CitationValidator and Deterministic Grounding."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import soundfile as sf

from src.agent.orchestrator import AuralQAgent
from src.agent.schemas import AgentResponse, EvidencePackage, GroundingReport
from src.agent.validator import CitationValidator, parse_numeric_candidate
from src.audio.validation import AudioMetadata
from src.llm.client import OllamaClient, OllamaResponse
from src.llm.schemas import IntentParseResult, ToolCallIntent
from src.tools.schemas import ToolResultModel


# ---------------------------------------------------------------------------
# Unit Tests: Token Extraction and Candidate Parsing
# ---------------------------------------------------------------------------

def test_parse_numeric_candidate():
    """Verify parsing of tokens into floats and normalized units."""
    val, unit = parse_numeric_candidate("1520.45 Hz")
    assert val == 1520.45
    assert unit == "hz"

    val, unit = parse_numeric_candidate("-12.4dBFS")
    assert val == -12.4
    assert unit == "dbfs"

    val, unit = parse_numeric_candidate("85%")
    assert val == 85.0
    assert unit == "%"

    val, unit = parse_numeric_candidate("22050")
    assert val == 22050.0
    assert unit == ""

    val, unit = parse_numeric_candidate("not_a_number")
    assert val is None


def test_extract_numeric_tokens():
    """Verify regex extraction of numbers from complex natural language responses."""
    validator = CitationValidator()
    text = (
        "The audio sample rate is 22050 Hz across 2.5 seconds. "
        "The spectral centroid was measured at 1520.5 Hz (approx 1.52 kHz), "
        "with an RMS energy of 0.082 (-12.4 dBFS). Energy rolloff is at 85%."
    )
    tokens = validator.extract_numeric_tokens(text)
    assert "22050 Hz" in tokens or "22050" in tokens
    assert "2.5 seconds" in tokens or "2.5" in tokens
    assert "1520.5 Hz" in tokens
    assert "1.52 kHz" in tokens
    assert "0.082" in tokens
    assert "-12.4 dBFS" in tokens
    assert "85%" in tokens


# ---------------------------------------------------------------------------
# Unit Tests: Grounding Verification Logic
# ---------------------------------------------------------------------------

def test_validator_exact_alias_and_tolerance_match():
    """Verify verification via exact citation aliases and tolerance matching."""
    validator = CitationValidator()
    evidence = EvidencePackage(
        query="What is the brightness?",
        status="SUCCESS",
        consolidated_metrics={
            "spectral_centroid.mean_centroid_hz": 1520.45,
            "spectral_rolloff.rolloff_hz": 3200.0,
        },
        consolidated_anchors={
            "spectral_centroid.perceived_brightness": "moderate",
        },
        all_citation_aliases={
            "spectral_centroid.mean_centroid_hz": [
                "1520.45", "1520.5", "1520", "1520.45 Hz", "1520.5 Hz", "1520 Hz", "1.52 kHz", "1.5 kHz"
            ],
            "spectral_rolloff.rolloff_hz": [
                "3200.0", "3200", "3200.0 Hz", "3200 Hz", "3.2 kHz"
            ],
        },
    )

    # Response with exact alias, rounded integer, and kHz conversion
    response = (
        "The recording has moderate brightness with a spectral centroid of 1520.5 Hz "
        "(or about 1.52 kHz). High-frequency energy rolls off at 3200 Hz."
    )
    report: GroundingReport = validator.validate(response, evidence)
    assert report.is_fully_grounded is True
    assert report.grounding_score == 1.0
    assert len(report.hallucinated_numbers) == 0
    assert len(report.verified_citations) == 3


def test_validator_detects_hallucinated_numbers():
    """Verify detection of fabricated metrics and ungrounded frequencies."""
    validator = CitationValidator()
    evidence = EvidencePackage(
        query="Is this audio loud?",
        status="SUCCESS",
        consolidated_metrics={
            "rms_energy.mean_rms": 0.082,
        },
        all_citation_aliases={
            "rms_energy.mean_rms": ["0.08", "0.082", "0.1"],
        },
    )

    # Response citing genuine 0.082, but fabricating 9500 Hz, 8 channels, and 72% distortion
    response = (
        "The audio has a mean RMS of 0.082, with an unexpected peak at 9500 Hz "
        "across 8 channels and 72% harmonic distortion."
    )
    report: GroundingReport = validator.validate(response, evidence)
    assert report.is_fully_grounded is False
    assert report.grounding_score < 1.0
    assert any("9500" in h for h in report.hallucinated_numbers)
    assert any("8" in h for h in report.hallucinated_numbers)
    assert any("72" in h for h in report.hallucinated_numbers)


def test_validator_query_echo_grounding():
    """Verify that repeating numbers from the user query is recognized as query echo, not hallucination."""
    validator = CitationValidator()
    evidence = EvidencePackage(
        query="Is the cutoff frequency near 1000 Hz?",
        status="SUCCESS",
        consolidated_metrics={
            "spectral_rolloff.rolloff_hz": 2400.0,
        },
        all_citation_aliases={
            "spectral_rolloff.rolloff_hz": ["2400", "2400 Hz", "2.4 kHz"],
        },
    )

    response = "No, the cutoff frequency is not 1000 Hz; the measured rolloff is 2400 Hz."
    report = validator.validate(response, evidence, query="Is the cutoff frequency near 1000 Hz?")
    assert report.is_fully_grounded is True
    assert len(report.hallucinated_numbers) == 0
    echo_matches = [m for m in report.verified_citations if m.match_type == "query_echo"]
    assert len(echo_matches) >= 1


def test_validator_abstention_and_qualitative_passes():
    """Verify that abstention or qualitative-only responses pass validation cleanly."""
    validator = CitationValidator()
    abstain_ev = EvidencePackage(query="Who is singing?", status="ABSTAIN")
    rep1 = validator.validate("I cannot answer identity questions.", abstain_ev)
    assert rep1.is_fully_grounded is True
    assert rep1.total_numbers_cited == 0

    success_ev = EvidencePackage(query="Is it bright?", status="SUCCESS")
    rep2 = validator.validate("The audio sounds dull and dark with no high-frequency content.", success_ev)
    assert rep2.is_fully_grounded is True
    assert rep2.total_numbers_cited == 0


def test_validator_deterministic_repair():
    """Verify that unverified numeric responses are replaced by deterministic evidence fallback."""
    validator = CitationValidator()
    evidence = EvidencePackage(
        query="Check brightness",
        status="SUCCESS",
        consolidated_metrics={"spectral_centroid.mean_centroid_hz": 3450.0},
        consolidated_anchors={"spectral_centroid.perceived_brightness": "bright"},
    )

    hallucinated_response = "The audio has a centroid of 99999 Hz."
    repaired_text, report = validator.repair_if_unverified(hallucinated_response, evidence)

    assert report.was_repaired is True
    assert report.original_response_text == hallucinated_response
    assert "99999" not in repaired_text
    assert "3450" in repaired_text
    assert "bright" in repaired_text
    assert "Verified DSP Grounding" in repaired_text


# ---------------------------------------------------------------------------
# Integration Tests: AuralQAgent with CitationValidator
# ---------------------------------------------------------------------------

def test_agent_with_hallucinated_response_triggers_repair():
    """Verify that AuralQAgent automatically repairs ungrounded responses generated by LLM."""
    mock_client = MagicMock(spec=OllamaClient)

    intent = IntentParseResult(
        user_intent="measure centroid",
        reasoning="spectral centroid measures brightness",
        selected_tools=[ToolCallIntent(tool_name="spectral_centroid", parameters={})],
        is_abstention=False,
    )
    mock_client.parse_intent.return_value = OllamaResponse(
        is_valid=True,
        status="success",
        parsed_intent=intent,
    )
    # LLM produces hallucinated numbers (e.g. 8888 Hz and 99%)
    mock_client.generate_text.return_value = {
        "response": "The brightness is 8888 Hz with 99% accuracy."
    }

    agent = AuralQAgent(client=mock_client)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1000 Hz sine
    sf.write(tmp_wav, sine, sr)

    try:
        resp: AgentResponse = agent.query("Is this sound bright?", audio_path=tmp_wav)
        assert resp.status == "success"
        assert resp.grounding_report is not None
        # Must have been repaired
        assert resp.grounding_report.was_repaired is True
        assert "8888" not in resp.response_text
        assert "Verified DSP Grounding" in resp.response_text
        assert resp.latency_breakdown["validation_ms"] >= 0.0
    finally:
        Path(tmp_wav).unlink(missing_ok=True)


def test_agent_with_grounded_response_passes_unmodified():
    """Verify that properly grounded LLM responses pass without substitution."""
    mock_client = MagicMock(spec=OllamaClient)

    intent = IntentParseResult(
        user_intent="measure energy",
        reasoning="rms energy",
        selected_tools=[ToolCallIntent(tool_name="rms_energy", parameters={})],
        is_abstention=False,
    )
    mock_client.parse_intent.return_value = OllamaResponse(
        is_valid=True,
        status="success",
        parsed_intent=intent,
    )
    # Grounded response using actual analytical sine RMS (0.5 peak / sqrt(2) ~= 0.354)
    mock_client.generate_text.return_value = {
        "response": "The audio has consistent volume with a measured mean RMS of 0.35."
    }

    agent = AuralQAgent(client=mock_client)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(tmp_wav, sine, sr)

    try:
        resp: AgentResponse = agent.query("What is the volume?", audio_path=tmp_wav)
        assert resp.status == "success"
        assert resp.grounding_report is not None
        assert resp.grounding_report.was_repaired is False
        assert "0.35" in resp.response_text
    finally:
        Path(tmp_wav).unlink(missing_ok=True)

