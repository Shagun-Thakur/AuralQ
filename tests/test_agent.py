"""Unit and integration test suite for Layer 4: AuralQAgent, SessionState, and ResponseSynthesizer."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import soundfile as sf

from src.agent.orchestrator import AuralQAgent
from src.agent.schemas import AgentResponse, EvidencePackage, TurnRecord
from src.agent.state import SessionState, serialize_tool_cache_key
from src.agent.synthesizer import ResponseSynthesizer
from src.audio.validation import AudioMetadata
from src.llm.client import OllamaClient, OllamaResponse
from src.llm.schemas import IntentParseResult, ToolCallIntent
from src.tools.schemas import ToolResultModel


# ---------------------------------------------------------------------------
# SessionState & Tool Caching Tests
# ---------------------------------------------------------------------------

def test_session_state_tool_caching_invariant_keys():
    """Verify that tool cache keys are invariant to parameter dictionary ordering."""
    state = SessionState()
    metadata = AudioMetadata(
        file_path="clip.wav",
        sample_rate=22050,
        num_channels=1,
        duration_sec=2.0,
        num_samples=44100,
        rms_dbfs=-12.0,
    )
    waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
    state.set_audio(metadata, waveform, 22050)

    res = ToolResultModel(
        tool="spectral_centroid",
        status="success",
        parameters={"n_fft": 2048, "hop_length": 512},
        result={"mean_centroid_hz": 1500.0},
        perceptual_anchors={"perceived_brightness": "moderate"},
        citation_aliases={},
        execution_time_ms=5.0,
    )

    # Cache with one key ordering
    state.cache_tool_result("spectral_centroid", {"n_fft": 2048, "hop_length": 512}, res)

    # Retrieve with opposite dictionary ordering
    cached = state.get_cached_tool_result("spectral_centroid", {"hop_length": 512, "n_fft": 2048})
    assert cached is not None
    assert cached.result["mean_centroid_hz"] == 1500.0


def test_session_state_cache_invalidation_on_audio_change():
    """Verify that loading a new audio file clears the tool cache."""
    state = SessionState()
    meta1 = AudioMetadata(
        file_path="clip1.wav",
        sample_rate=22050,
        num_channels=1,
        duration_sec=1.0,
        num_samples=22050,
        rms_dbfs=-15.0,
    )
    meta2 = AudioMetadata(
        file_path="clip2.wav",
        sample_rate=22050,
        num_channels=1,
        duration_sec=1.0,
        num_samples=22050,
        rms_dbfs=-15.0,
    )
    wf = np.zeros(22050)

    state.set_audio(meta1, wf, 22050)
    res = ToolResultModel(
        tool="rms_energy",
        status="success",
        parameters={},
        result={"mean_rms": 0.5},
        perceptual_anchors={},
        citation_aliases={},
        execution_time_ms=2.0,
    )
    state.cache_tool_result("rms_energy", {}, res)
    assert state.get_cached_tool_result("rms_energy", {}) is not None

    # Load audio 2 -> cache must be invalidated
    state.set_audio(meta2, wf, 22050)
    assert state.get_cached_tool_result("rms_energy", {}) is None


def test_session_state_history_context():
    """Verify multi-turn history tracking and compact context generation."""
    state = SessionState()
    ev = EvidencePackage(query="dummy", status="SUCCESS")

    turn1 = TurnRecord(
        turn_id=1,
        timestamp=time.time(),
        query="Is this loud?",
        evidence=ev,
        response_text="The audio has moderate volume.",
    )
    turn2 = TurnRecord(
        turn_id=2,
        timestamp=time.time(),
        query="Is it bright?",
        evidence=ev,
        response_text="The audio has bright timbre.",
    )
    state.add_turn(turn1)
    state.add_turn(turn2)

    ctx = state.get_recent_history_context(max_turns=2)
    assert "User: Is this loud?" in ctx
    assert "AuralQ: The audio has moderate volume." in ctx
    assert "User: Is it bright?" in ctx


# ---------------------------------------------------------------------------
# ResponseSynthesizer Tests
# ---------------------------------------------------------------------------

def test_synthesizer_abstention_fast_path():
    """Verify that abstention bypasses LLM call and returns immediate explanation."""
    synth = ResponseSynthesizer()
    ev = EvidencePackage(
        query="Who wrote this song?",
        status="ABSTAIN",
        abstention_reason="Cannot identify artists or musical lyrics.",
    )
    text, lat_ms = synth.synthesize("Who wrote this song?", ev)
    assert "outside the scope of acoustic signal processing" in text
    assert "Cannot identify artists" in text
    assert lat_ms == 0.0


def test_synthesizer_deterministic_fallback():
    """Verify that deterministic fallback properly discloses metrics, anchors, and warnings."""
    synth = ResponseSynthesizer()
    ev = EvidencePackage(
        query="How loud is this?",
        status="SUCCESS",
        consolidated_metrics={"rms_energy.mean_rms": 0.082},
        consolidated_anchors={"rms_energy.loudness_consistency": "consistent"},
        warnings=["Low signal energy detected"],
    )
    text = synth._deterministic_fallback_synthesis(ev)
    assert "consistent" in text
    assert "0.082" in text
    assert "Low signal energy detected" in text


# ---------------------------------------------------------------------------
# AuralQAgent Orchestration Tests
# ---------------------------------------------------------------------------

def test_agent_query_with_mocked_llm_and_cache_hit():
    """Verify full agent turn execution and cache hit on subsequent identical tool call."""
    mock_client = MagicMock(spec=OllamaClient)

    # Turn 1 Intent: call rms_energy
    turn1_intent = IntentParseResult(
        user_intent="measure loudness",
        reasoning="RMS energy measures volume consistency",
        selected_tools=[ToolCallIntent(tool_name="rms_energy", parameters={})],
        is_abstention=False,
    )
    mock_client.parse_intent.return_value = OllamaResponse(
        is_valid=True,
        status="success",
        parsed_intent=turn1_intent,
    )
    mock_client.generate_text.return_value = {"response": "The audio has consistent loudness with mean RMS of 0.707."}

    agent = AuralQAgent(client=mock_client)

    # Create synthetic test file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(tmp_wav, sine, sr)

    try:
        # Turn 1
        resp1: AgentResponse = agent.query("Is this audio loud?", audio_path=tmp_wav)
        assert resp1.status == "success"
        assert "rms_energy" in resp1.evidence.tools_called
        assert len(resp1.cached_tools_used) == 0
        assert resp1.latency_breakdown["dsp_ms"] > 0

        # Turn 2: Ask similar question requiring same tool
        resp2: AgentResponse = agent.query("What is the exact RMS energy?")
        assert resp2.status == "success"
        # Tool result should be retrieved from cache
        assert "rms_energy" in resp2.cached_tools_used
        assert resp2.latency_breakdown["dsp_ms"] == 0.0
        assert len(agent.session_state.turns) == 2

    finally:
        Path(tmp_wav).unlink(missing_ok=True)


def test_agent_abstention_flow():
    """Verify that an out-of-scope query cleanly returns abstained status and zero DSP calls."""
    mock_client = MagicMock(spec=OllamaClient)
    abstain_intent = IntentParseResult(
        user_intent="out of scope question",
        reasoning="Medical diagnosis is impossible via basic DSP",
        selected_tools=[],
        is_abstention=True,
        abstention_reason="Cannot diagnose medical ailments from audio.",
    )
    mock_client.parse_intent.return_value = OllamaResponse(
        is_valid=True,
        status="abstained",
        parsed_intent=abstain_intent,
    )

    agent = AuralQAgent(client=mock_client)
    resp = agent.query("Does this cough sound like pneumonia?")
    assert resp.status == "abstained"
    assert resp.evidence.status == "ABSTAIN"
    assert resp.latency_breakdown["dsp_ms"] == 0.0
    assert "Cannot diagnose medical ailments" in resp.response_text


def test_agent_dataset_profiler_flow():
    """Verify end-to-end dataset loading and dataset_profiler tool execution."""
    mock_client = MagicMock(spec=OllamaClient)
    dataset_intent = IntentParseResult(
        user_intent="profile dataset",
        reasoning="dataset audit requested",
        selected_tools=[ToolCallIntent(tool_name="dataset_profiler", parameters={})],
        is_abstention=False,
    )
    mock_client.parse_intent.return_value = OllamaResponse(
        is_valid=True,
        status="success",
        parsed_intent=dataset_intent,
    )
    mock_client.generate_text.return_value = {"response": "The dataset contains 2 files and requires resampling."}

    agent = AuralQAgent(client=mock_client)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir)
        sf.write(str(dir_path / "clip1.wav"), np.sin(np.linspace(0, 10, 8000)), 8000)
        sf.write(str(dir_path / "clip2.wav"), np.sin(np.linspace(0, 10, 16000)), 16000)

        resp = agent.query("Audit my dataset directory", dataset_path=dir_path)
        assert resp.status == "success"
        assert "dataset_profiler" in resp.evidence.tools_called
        assert "dataset_profiler.total_files" in resp.evidence.consolidated_metrics
        assert resp.evidence.consolidated_metrics["dataset_profiler.total_files"] == 2


def test_agent_ingestion_warning_propagation():
    """Verify that clipping warnings from Layer 1 ingestion propagate into evidence and disclosures."""
    mock_client = MagicMock(spec=OllamaClient)
    intent = IntentParseResult(
        user_intent="measure energy",
        reasoning="check volume",
        selected_tools=[ToolCallIntent(tool_name="rms_energy", parameters={})],
        is_abstention=False,
    )
    mock_client.parse_intent.return_value = OllamaResponse(
        is_valid=True,
        status="success",
        parsed_intent=intent,
    )
    # Synthesizer fallback used when generate_text throws
    mock_client.generate_text.side_effect = Exception("Fallback test")

    agent = AuralQAgent(client=mock_client)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = 2.0 * np.sin(2 * np.pi * 440 * t)
    clipped_wave = np.clip(sine, -1.0, 1.0)  # Saturated alternating sine -> triggers clipping warning without DC bias
    sf.write(tmp_wav, clipped_wave, sr)

    try:
        resp = agent.query("Check the volume", audio_path=tmp_wav)
        assert resp.status == "success"
        assert any("clipping" in w.lower() for w in resp.evidence.warnings)
        assert "clipping" in resp.response_text.lower()
    finally:
        Path(tmp_wav).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Live Ollama Multi-Turn Agent Test
# ---------------------------------------------------------------------------

@pytest.mark.live_ollama
def test_live_ollama_agent_multiturn():
    """Test full multi-turn conversational session with live local Ollama instance."""
    # Pre-flight check
    try:
        import requests
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("Local Ollama server is not running on 127.0.0.1:11434")
    except Exception:
        pytest.skip("Local Ollama server unreachable")

    agent = AuralQAgent()

    # Generate synthetic 440 Hz pure sine wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = 0.6 * np.sin(2 * np.pi * 440 * t)
    sf.write(tmp_wav, sine, sr)

    try:
        # Turn 1: Ask about brightness
        resp1 = agent.query("Does this recording sound bright or dull?", audio_path=tmp_wav)
        assert resp1.status == "success"
        assert "spectral_centroid" in resp1.evidence.tools_called
        assert len(resp1.response_text) > 10

        # Turn 2: Follow-up asking about exact centroid (should hit cache)
        resp2 = agent.query("What is the exact spectral center frequency of this audio?")
        assert resp2.status == "success"
        assert "spectral_centroid" in resp2.cached_tools_used
        assert resp2.latency_breakdown["dsp_ms"] == 0.0

    finally:
        Path(tmp_wav).unlink(missing_ok=True)
