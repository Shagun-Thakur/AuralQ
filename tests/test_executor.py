"""Unit and integration tests for Layer 4: Tool Execution Engine and Evidence Packaging."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.agent.executor import ToolExecutor
from src.agent.schemas import EvidencePackage
from src.audio.loader import load_audio_file
from src.audio.validation import AudioMetadata
from src.llm.client import OllamaClient
from src.llm.schemas import IntentParseResult, ToolCallIntent


@pytest.fixture
def synthetic_sine_1000hz():
    """Generate 2 seconds of pure 1000 Hz sine wave at A=0.5, SR=22050."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    return waveform, sr


@pytest.fixture
def synthetic_dataset_dir(tmp_path):
    """Generate a minimal 3-file dataset for directory tool testing."""
    d = tmp_path / "test_corpus"
    d.mkdir()
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(d / "sample1.wav"), y, sr)
    sf.write(str(d / "sample2.wav"), y, sr)
    return d


# ---------------------------------------------------------------------------
# 1. Deterministic Unit Tests
# ---------------------------------------------------------------------------

def test_executor_single_tool(synthetic_sine_1000hz):
    """Verify single-tool execution and evidence packaging."""
    waveform, sr = synthetic_sine_1000hz
    executor = ToolExecutor()

    intent = IntentParseResult(
        user_intent="brightness_analysis",
        reasoning="Spectral centroid corresponds to perceived brightness.",
        selected_tools=[
            ToolCallIntent(tool_name="spectral_centroid", parameters={"n_fft": 2048})
        ],
    )

    meta = AudioMetadata(
        file_path="synthetic_sine.wav",
        duration_sec=2.0,
        sample_rate=sr,
        num_channels=1,
        num_samples=len(waveform),
        rms_dbfs=-6.02,
    )

    evidence: EvidencePackage = executor.execute_intent(
        intent_result=intent,
        waveform=waveform,
        sr=sr,
        audio_metadata=meta,
        query="Does this recording sound bright?",
    )

    assert evidence.status == "SUCCESS"
    assert evidence.tools_called == ["spectral_centroid"]
    assert len(evidence.tool_results) == 1
    assert "mean_hz" in evidence.consolidated_metrics
    assert np.isclose(evidence.consolidated_metrics["mean_hz"], 1000.0, atol=25.0)
    assert "perceived_brightness" in evidence.consolidated_anchors
    assert len(evidence.all_citation_aliases["mean_hz"]) > 0

    # Test evidence text formatting
    formatted = evidence.format_evidence_for_llm()
    assert "<evidence>" in formatted
    assert "STATUS: SUCCESS" in formatted
    assert "mean_hz: 1000" in formatted or "1000." in formatted


def test_executor_multi_tool(synthetic_sine_1000hz):
    """Verify concurrent multi-tool execution and namespaced metric packaging."""
    waveform, sr = synthetic_sine_1000hz
    executor = ToolExecutor()

    intent = IntentParseResult(
        user_intent="brightness_and_loudness",
        reasoning="Testing multi-tool execution.",
        selected_tools=[
            ToolCallIntent(tool_name="spectral_centroid", parameters={}),
            ToolCallIntent(tool_name="rms_energy", parameters={}),
        ],
    )

    evidence = executor.execute_intent(
        intent_result=intent,
        waveform=waveform,
        sr=sr,
        query="Is this audio bright and consistently loud?",
    )

    assert evidence.status == "SUCCESS"
    assert len(evidence.tools_called) == 2
    assert "spectral_centroid.mean_hz" in evidence.consolidated_metrics
    assert "rms_energy.mean_rms" in evidence.consolidated_metrics
    assert "rms_energy.loudness_stability" in evidence.consolidated_anchors


def test_executor_dataset_profiler_routing(synthetic_dataset_dir):
    """Verify tool execution routing for dataset_profiler."""
    executor = ToolExecutor()

    intent = IntentParseResult(
        user_intent="dataset_audit",
        reasoning="Auditing corpus.",
        selected_tools=[
            ToolCallIntent(tool_name="dataset_profiler", parameters={"dataset_path": str(synthetic_dataset_dir)})
        ],
    )

    evidence = executor.execute_intent(
        intent_result=intent,
        dataset_path=str(synthetic_dataset_dir),
        query="Audit my dataset.",
    )

    assert evidence.status == "SUCCESS"
    assert evidence.tools_called == ["dataset_profiler"]
    assert evidence.consolidated_metrics["total_files"] == 2
    assert "recommendations" in evidence.consolidated_metrics


def test_executor_abstention_passthrough():
    """Verify that abstention intents bypass DSP execution and return clean evidence."""
    executor = ToolExecutor()

    intent = IntentParseResult(
        user_intent="out_of_scope",
        reasoning="Question is about world history.",
        selected_tools=[],
        is_abstention=True,
        abstention_reason="Cannot answer history questions via audio DSP.",
    )

    evidence = executor.execute_intent(
        intent_result=intent,
        query="Who was president in 1865?",
    )

    assert evidence.status == "ABSTAIN"
    assert "Cannot answer history questions" in evidence.abstention_reason
    assert len(evidence.tools_called) == 0

    formatted = evidence.format_evidence_for_llm()
    assert "STATUS: ABSTAIN" in formatted


def test_executor_warning_propagation(synthetic_sine_1000hz):
    """Verify that Layer 1 ingestion warnings propagate into the EvidencePackage."""
    waveform, sr = synthetic_sine_1000hz
    executor = ToolExecutor()

    intent = IntentParseResult(
        user_intent="rolloff",
        reasoning="Checking rolloff.",
        selected_tools=[ToolCallIntent(tool_name="spectral_rolloff", parameters={})],
    )

    meta = AudioMetadata(
        file_path="clipped_audio.wav",
        duration_sec=2.0,
        sample_rate=sr,
        num_channels=1,
        num_samples=len(waveform),
        rms_dbfs=-2.0,
        is_clipped=True,
        clipping_ratio=0.08,
        warnings=["CLIPPING_WARNING: 8.0% of samples exhibit flat-topped clipping."],
    )

    evidence = executor.execute_intent(
        intent_result=intent,
        waveform=waveform,
        sr=sr,
        audio_metadata=meta,
    )

    assert evidence.status == "SUCCESS"
    assert any("CLIPPING_WARNING" in w for w in evidence.warnings)
    formatted = evidence.format_evidence_for_llm()
    assert "INGESTION_WARNINGS: CLIPPING_WARNING" in formatted


# ---------------------------------------------------------------------------
# 2. End-to-End Integration Test (Ollama -> Executor -> DSP)
# ---------------------------------------------------------------------------

def is_ollama_reachable() -> bool:
    import requests
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not is_ollama_reachable(), reason="Local Ollama server is not running")
def test_live_e2e_pipeline(synthetic_sine_1000hz):
    """
    Test full vertical pipeline:
    Query -> Ollama Intent Parser -> Tool Executor -> Deterministic DSP -> Evidence Package.
    """
    waveform, sr = synthetic_sine_1000hz
    client = OllamaClient()
    executor = ToolExecutor(ollama_client=client)

    question = "Does this audio sound bright or dull?"
    llm_res = client.parse_intent(question)

    assert llm_res.is_valid is True
    assert llm_res.parsed_intent is not None

    evidence = executor.execute_intent(
        intent_result=llm_res.parsed_intent,
        waveform=waveform,
        sr=sr,
        query=question,
    )

    assert evidence.status == "SUCCESS"
    assert "spectral_centroid" in evidence.tools_called
    # Check that computed centroid of 1000 Hz sine is verified
    assert "mean_hz" in evidence.consolidated_metrics
    assert np.isclose(evidence.consolidated_metrics["mean_hz"], 1000.0, atol=25.0)
    assert evidence.consolidated_anchors["perceived_brightness"] == "dull"

