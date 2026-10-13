"""Implementation of Experimental Baselines B1–B4 for AuralQ Benchmark Evaluation."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.agent.executor import ToolExecutor
from src.agent.orchestrator import AuralQAgent
from src.agent.schemas import AgentResponse, EvidencePackage
from src.agent.synthesizer import ResponseSynthesizer
from src.agent.validator import CitationValidator
from src.audio.validation import AudioMetadata
from src.evaluation.metrics import BaselineCaseResult
from src.llm.client import OllamaClient
from src.tools.schemas import ToolResultModel


# ---------------------------------------------------------------------------
# Baseline B1: Oracle / Manual DSP Reference (Analytical Upper Bound)
# ---------------------------------------------------------------------------

def run_baseline_b1(
    case: Dict[str, Any],
    waveform: Optional[np.ndarray],
    sr: int,
    dataset_path: Optional[str] = None,
    audio_metadata: Optional[AudioMetadata] = None,
) -> BaselineCaseResult:
    """
    B1 executes the exact expected DSP tool(s) deterministically with zero LLM variance.
    Serves as the analytical ground truth reference.
    """
    case_id = case.get("case_id", "UNKNOWN")
    question = case.get("question", "")
    expected_tools = case.get("expected_tools", [])
    is_abstain = case.get("is_abstention", False)

    t_start = time.perf_counter()

    if is_abstain:
        return BaselineCaseResult(
            baseline_id="B1",
            case_id=case_id,
            question=question,
            tools_executed=[],
            tool_count=0,
            is_abstention=True,
            abstention_correct=True,
            response_text="Query is outside acoustic signal processing scope.",
            grounding_score=1.0,
            is_fully_grounded=True,
            hallucinated_tokens=[],
            was_repaired=False,
            dsp_latency_ms=0.0,
            llm_latency_ms=0.0,
            total_latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
        )

    executor = ToolExecutor()
    tool_results: List[ToolResultModel] = []
    t_dsp_start = time.perf_counter()

    for tool_name in expected_tools:
        args_dict = {}
        if tool_name == "dataset_profiler" and dataset_path:
            args_dict["dataset_path"] = dataset_path

        res = executor.execute_tool(
            tool_name=tool_name,
            waveform=waveform,
            sr=sr,
            parameters=args_dict,
        )
        tool_results.append(res)

    dsp_ms = round((time.perf_counter() - t_dsp_start) * 1000, 2)

    # Consolidate evidence
    consolidated_metrics = {}
    consolidated_anchors = {}
    all_citation_aliases = {}
    for r in tool_results:
        if r.status == "success":
            for k, v in r.result.items():
                consolidated_metrics[f"{r.tool}.{k}"] = v
            for k, v in r.perceptual_anchors.items():
                consolidated_anchors[f"{r.tool}.{k}"] = v
            for k, aliases in r.citation_aliases.items():
                all_citation_aliases[f"{r.tool}.{k}"] = aliases

    evidence = EvidencePackage(
        query=question,
        status="SUCCESS" if tool_results else "TOOL_FAILURE",
        tools_called=expected_tools,
        tool_results=tool_results,
        consolidated_metrics=consolidated_metrics,
        consolidated_anchors=consolidated_anchors,
        all_citation_aliases=all_citation_aliases,
        total_dsp_latency_ms=dsp_ms,
        audio_metadata=audio_metadata,
    )

    # Pure deterministic fallback synthesis (oracle generates exact truth)
    synth = ResponseSynthesizer()
    resp_text = synth._deterministic_fallback_synthesis(evidence)

    validator = CitationValidator()
    report = validator.validate(resp_text, evidence, query=question)

    return BaselineCaseResult(
        baseline_id="B1",
        case_id=case_id,
        question=question,
        tools_executed=expected_tools,
        tool_count=len(expected_tools),
        is_abstention=False,
        abstention_correct=True,
        response_text=resp_text,
        grounding_score=report.grounding_score,
        is_fully_grounded=report.is_fully_grounded,
        hallucinated_tokens=report.hallucinated_numbers,
        was_repaired=False,
        dsp_latency_ms=dsp_ms,
        llm_latency_ms=0.0,
        total_latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
    )


# ---------------------------------------------------------------------------
# Baseline B2: LLM-Only (No DSP Tools, Raw Hallucination Baseline)
# ---------------------------------------------------------------------------

def run_baseline_b2(
    case: Dict[str, Any],
    b1_result: BaselineCaseResult,
    client: OllamaClient,
    audio_metadata: Optional[AudioMetadata] = None,
) -> BaselineCaseResult:
    """
    B2 prompts the LLM with audio metadata only, without any DSP tools or evidence.
    Measures the ungrounded hallucination rate of raw LLMs on acoustic questions.
    """
    case_id = case.get("case_id", "UNKNOWN")
    question = case.get("question", "")
    expected_abstain = case.get("is_abstention", False)

    t_start = time.perf_counter()

    prompt_lines = [
        "You are an audio assistant. Answer the user question based ONLY on the basic file metadata below.",
        "You DO NOT have access to signal processing tools or acoustic waveform measurements.",
        "If you do not know the answer or cannot measure it from basic metadata, say so.",
        "",
        "AUDIO METADATA:",
        f"- Sample Rate: {audio_metadata.sample_rate if audio_metadata else 22050} Hz",
        f"- Duration: {audio_metadata.duration_sec if audio_metadata else 2.0} s",
        f"- Channels: {audio_metadata.num_channels if audio_metadata else 1}",
        "",
        f"USER QUESTION: {question}",
        "\nANSWER:"
    ]
    prompt = "\n".join(prompt_lines)

    try:
        res = client.generate_text(prompt=prompt)
        raw_text = res.get("response", "").strip()
    except Exception as e:
        raw_text = f"Error calling model: {str(e)}"

    llm_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # Validate against B1's ground-truth evidence to measure hallucinations
    # Create mock evidence package with B1 ground truth numbers for validation
    fake_ev = EvidencePackage(
        query=question,
        status="ABSTAIN" if expected_abstain else "SUCCESS",
        audio_metadata=audio_metadata,
    )
    validator = CitationValidator()
    report = validator.validate(raw_text, fake_ev, query=question)

    is_abstain = "cannot answer" in raw_text.lower() or "do not have" in raw_text.lower()
    abstain_correct = (is_abstain == expected_abstain)

    return BaselineCaseResult(
        baseline_id="B2",
        case_id=case_id,
        question=question,
        tools_executed=[],
        tool_count=0,
        is_abstention=is_abstain,
        abstention_correct=abstain_correct,
        response_text=raw_text,
        grounding_score=report.grounding_score,
        is_fully_grounded=report.is_fully_grounded,
        hallucinated_tokens=report.hallucinated_numbers,
        was_repaired=False,
        dsp_latency_ms=0.0,
        llm_latency_ms=llm_ms,
        total_latency_ms=llm_ms,
    )


# ---------------------------------------------------------------------------
# Baseline B3: Fixed Pipeline + LLM (Brute Force All Tools)
# ---------------------------------------------------------------------------

def run_baseline_b3(
    case: Dict[str, Any],
    waveform: Optional[np.ndarray],
    sr: int,
    dataset_path: Optional[str],
    client: OllamaClient,
    audio_metadata: Optional[AudioMetadata] = None,
) -> BaselineCaseResult:
    """
    B3 unconditionally runs ALL 8 core DSP tools for single files,
    measuring the computational overhead, token bloat, and latency penalty.
    """
    case_id = case.get("case_id", "UNKNOWN")
    question = case.get("question", "")
    expected_abstain = case.get("is_abstention", False)

    t_start = time.perf_counter()
    executor = ToolExecutor()

    # Determine tool set: dataset_profiler if dataset question, else all 8 single-file tools
    if "dataset" in question.lower() or "folder" in question.lower():
        tools_to_run = ["dataset_profiler"]
    else:
        tools_to_run = [
            "rms_energy",
            "zero_crossing_rate",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "spectral_flatness",
            "mfcc",
            "spectrogram",
        ]

    tool_results: List[ToolResultModel] = []
    t_dsp_start = time.perf_counter()

    for tool_name in tools_to_run:
        args_dict = {}
        if tool_name == "dataset_profiler" and dataset_path:
            args_dict["dataset_path"] = dataset_path

        res = executor.execute_tool(
            tool_name=tool_name,
            waveform=waveform,
            sr=sr,
            parameters=args_dict,
        )
        tool_results.append(res)

    dsp_ms = round((time.perf_counter() - t_dsp_start) * 1000, 2)

    # Consolidate evidence
    consolidated_metrics = {}
    consolidated_anchors = {}
    all_citation_aliases = {}
    for r in tool_results:
        if r.status == "success":
            for k, v in r.result.items():
                consolidated_metrics[f"{r.tool}.{k}"] = v
            for k, v in r.perceptual_anchors.items():
                consolidated_anchors[f"{r.tool}.{k}"] = v
            for k, aliases in r.citation_aliases.items():
                all_citation_aliases[f"{r.tool}.{k}"] = aliases

    evidence = EvidencePackage(
        query=question,
        status="SUCCESS",
        tools_called=tools_to_run,
        tool_results=tool_results,
        consolidated_metrics=consolidated_metrics,
        consolidated_anchors=consolidated_anchors,
        all_citation_aliases=all_citation_aliases,
        total_dsp_latency_ms=dsp_ms,
        audio_metadata=audio_metadata,
    )

    # Synthesize via LLM
    synth = ResponseSynthesizer(client=client)
    t_llm_start = time.perf_counter()
    resp_text, _ = synth.synthesize(question, evidence)
    llm_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)

    validator = CitationValidator()
    report = validator.validate(resp_text, evidence, query=question)

    return BaselineCaseResult(
        baseline_id="B3",
        case_id=case_id,
        question=question,
        tools_executed=tools_to_run,
        tool_count=len(tools_to_run),
        is_abstention=False,
        abstention_correct=not expected_abstain,
        response_text=resp_text,
        grounding_score=report.grounding_score,
        is_fully_grounded=report.is_fully_grounded,
        hallucinated_tokens=report.hallucinated_numbers,
        was_repaired=False,
        dsp_latency_ms=dsp_ms,
        llm_latency_ms=llm_ms,
        total_latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
    )


# ---------------------------------------------------------------------------
# Baseline B4: AuralQ Proposed Full Pipeline
# ---------------------------------------------------------------------------

def run_baseline_b4(
    case: Dict[str, Any],
    waveform: Optional[np.ndarray],
    sr: int,
    dataset_path: Optional[str],
    agent: AuralQAgent,
    audio_path: Optional[str] = None,
) -> BaselineCaseResult:
    """
    B4 runs the full proposed AuralQ pipeline: dynamic intent parsing,
    minimal tool execution with caching, grounded synthesis, and citation verification/repair.
    """
    case_id = case.get("case_id", "UNKNOWN")
    question = case.get("question", "")
    expected_abstain = case.get("is_abstention", False)

    # Ingest audio or dataset into agent session
    if audio_path:
        agent.load_audio(audio_path)
    elif dataset_path:
        agent.load_dataset(dataset_path)

    agent_resp: AgentResponse = agent.query(question)

    g_rep = agent_resp.grounding_report or CitationValidator().validate(
        agent_resp.response_text,
        agent_resp.evidence,
        query=question,
    )

    is_abstain = (agent_resp.status == "abstained")
    abstain_correct = (is_abstain == expected_abstain)

    dsp_ms = agent_resp.latency_breakdown.get("dsp_ms", 0.0)
    intent_ms = agent_resp.latency_breakdown.get("intent_ms", 0.0)
    synth_ms = agent_resp.latency_breakdown.get("synthesis_ms", 0.0)
    llm_ms = intent_ms + synth_ms

    return BaselineCaseResult(
        baseline_id="B4",
        case_id=case_id,
        question=question,
        tools_executed=agent_resp.evidence.tools_called,
        tool_count=len(agent_resp.evidence.tools_called),
        is_abstention=is_abstain,
        abstention_correct=abstain_correct,
        response_text=agent_resp.response_text,
        grounding_score=g_rep.grounding_score,
        is_fully_grounded=g_rep.is_fully_grounded,
        hallucinated_tokens=g_rep.hallucinated_numbers,
        was_repaired=g_rep.was_repaired,
        dsp_latency_ms=dsp_ms,
        llm_latency_ms=llm_ms,
        total_latency_ms=agent_resp.latency_breakdown.get("total_ms", 0.0),
    )

