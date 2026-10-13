"""Unit tests for Milestone E6: Evaluation Framework, Baselines B1–B4, and Metrics."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import soundfile as sf

from src.agent.orchestrator import AuralQAgent
from src.evaluation.baselines import (
    run_baseline_b1,
    run_baseline_b2,
    run_baseline_b3,
    run_baseline_b4,
)
from src.evaluation.metrics import (
    BaselineCaseResult,
    BaselineSummary,
    BenchmarkComparisonReport,
    compute_baseline_summary,
)
from src.llm.client import OllamaClient, OllamaResponse
from src.llm.schemas import IntentParseResult, ToolCallIntent


# ---------------------------------------------------------------------------
# Unit Tests: Metrics Computation & Markdown Table
# ---------------------------------------------------------------------------

def test_compute_baseline_summary_statistics():
    """Verify aggregated metrics math across baseline case results."""
    cases = [
        BaselineCaseResult(
            baseline_id="B4",
            case_id="BC_001",
            question="Is it bright?",
            tools_executed=["spectral_centroid"],
            tool_count=1,
            is_abstention=False,
            abstention_correct=True,
            response_text="The centroid is 1500 Hz.",
            grounding_score=1.0,
            is_fully_grounded=True,
            hallucinated_tokens=[],
            was_repaired=False,
            dsp_latency_ms=5.0,
            llm_latency_ms=3000.0,
            total_latency_ms=3005.0,
        ),
        BaselineCaseResult(
            baseline_id="B4",
            case_id="BC_002",
            question="Is it loud?",
            tools_executed=["rms_energy"],
            tool_count=1,
            is_abstention=False,
            abstention_correct=True,
            response_text="The audio has RMS of 99.0.",  # Hallucinated
            grounding_score=0.0,
            is_fully_grounded=False,
            hallucinated_tokens=["99.0"],
            was_repaired=True,
            dsp_latency_ms=2.0,
            llm_latency_ms=2500.0,
            total_latency_ms=2502.0,
        ),
    ]

    summary = compute_baseline_summary("B4", "AuralQ Proposed", cases)
    assert summary.total_cases == 2
    assert summary.avg_tools_per_query == 1.0
    assert summary.abstention_accuracy == 100.0
    assert summary.citation_grounding_rate == 50.0
    assert summary.numeric_hallucination_rate == 50.0
    assert summary.repair_rate == 50.0
    assert summary.avg_dsp_latency_ms == 3.5
    assert summary.avg_total_latency_ms == 2753.5


def test_benchmark_comparison_report_markdown_generation():
    """Verify conversion of baseline summaries to markdown table."""
    b1_summary = BaselineSummary(
        baseline_id="B1",
        baseline_name="Oracle",
        total_cases=2,
        avg_tools_per_query=1.0,
        abstention_accuracy=100.0,
        citation_grounding_rate=100.0,
        numeric_hallucination_rate=0.0,
        avg_dsp_latency_ms=5.0,
        avg_llm_latency_ms=0.0,
        avg_total_latency_ms=5.0,
        repair_rate=0.0,
    )
    b4_summary = BaselineSummary(
        baseline_id="B4",
        baseline_name="AuralQ",
        total_cases=2,
        avg_tools_per_query=1.2,
        abstention_accuracy=100.0,
        citation_grounding_rate=100.0,
        numeric_hallucination_rate=0.0,
        avg_dsp_latency_ms=10.0,
        avg_llm_latency_ms=3500.0,
        avg_total_latency_ms=3510.0,
        repair_rate=0.0,
    )

    report = BenchmarkComparisonReport(baselines={"B1": b1_summary, "B4": b4_summary})
    table = report.to_markdown_table()
    assert "| **Citation Grounding Rate** |" in table
    assert "100.0%" in table
    assert "| **Avg Tools Executed / Query** |" in table


# ---------------------------------------------------------------------------
# Unit Tests: Individual Baseline Runners
# ---------------------------------------------------------------------------

def test_run_baseline_b1_oracle():
    """Verify B1 runs ground-truth tool deterministically."""
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    case = {
        "case_id": "BC_002",
        "question": "Does this recording sound bright or dull?",
        "expected_tools": ["spectral_centroid"],
        "is_abstention": False,
    }

    res: BaselineCaseResult = run_baseline_b1(case, sine, sr)
    assert res.baseline_id == "B1"
    assert res.tools_executed == ["spectral_centroid"]
    assert res.tool_count == 1
    assert res.is_fully_grounded is True
    assert res.grounding_score == 1.0
    assert res.dsp_latency_ms > 0
    assert res.llm_latency_ms == 0.0


def test_run_baseline_b1_abstention():
    """Verify B1 correctly abstains without executing DSP."""
    case = {
        "case_id": "BC_019",
        "question": "What is the capital of France?",
        "expected_tools": [],
        "is_abstention": True,
    }
    res = run_baseline_b1(case, None, 22050)
    assert res.is_abstention is True
    assert res.abstention_correct is True
    assert res.tool_count == 0


def test_run_baseline_b2_llm_only_catches_hallucination():
    """Verify B2 catches ungrounded numbers when LLM attempts to answer without DSP."""
    mock_client = MagicMock(spec=OllamaClient)
    # LLM hallucinates an exact centroid frequency of 3850 Hz without tool evidence
    mock_client.generate_text.return_value = {
        "response": "Based on the audio file, the brightness is high with a spectral centroid of 3850 Hz."
    }

    case = {
        "case_id": "BC_002",
        "question": "Does this recording sound bright or dull?",
        "expected_tools": ["spectral_centroid"],
        "is_abstention": False,
    }

    res = run_baseline_b2(case, None, mock_client)
    assert res.baseline_id == "B2"
    assert res.tool_count == 0
    assert res.is_fully_grounded is False
    assert any("3850" in h for h in res.hallucinated_tokens)


def test_run_baseline_b3_fixed_pipeline_runs_all_tools():
    """Verify B3 unconditionally executes all 8 tools."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.generate_text.return_value = {
        "response": "The audio has high brightness."
    }

    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    case = {
        "case_id": "BC_002",
        "question": "Does this sound bright?",
        "expected_tools": ["spectral_centroid"],
        "is_abstention": False,
    }

    res = run_baseline_b3(case, sine, sr, None, mock_client)
    assert res.baseline_id == "B3"
    assert res.tool_count == 8
    assert "spectrogram" in res.tools_executed
    assert "mfcc" in res.tools_executed
    assert "rms_energy" in res.tools_executed


def test_run_baseline_b4_auralq_dynamic():
    """Verify B4 executes only minimal sufficient tools via agent loop."""
    mock_client = MagicMock(spec=OllamaClient)

    intent = IntentParseResult(
        user_intent="measure brightness",
        reasoning="spectral centroid",
        selected_tools=[ToolCallIntent(tool_name="spectral_centroid", parameters={})],
        is_abstention=False,
    )
    mock_client.parse_intent.return_value = OllamaResponse(
        is_valid=True,
        status="success",
        parsed_intent=intent,
    )
    mock_client.generate_text.return_value = {
        "response": "The audio has moderate brightness."
    }

    agent = AuralQAgent(client=mock_client)

    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    sf.write(tmp_wav, sine, sr)

    try:
        case = {
            "case_id": "BC_002",
            "question": "Does this recording sound bright or dull?",
            "expected_tools": ["spectral_centroid"],
            "is_abstention": False,
        }
        res = run_baseline_b4(case, sine, sr, None, agent, audio_path=tmp_wav)
        assert res.baseline_id == "B4"
        assert res.tool_count == 1
        assert res.tools_executed == ["spectral_centroid"]
        assert res.is_fully_grounded is True
    finally:
        Path(tmp_wav).unlink(missing_ok=True)

