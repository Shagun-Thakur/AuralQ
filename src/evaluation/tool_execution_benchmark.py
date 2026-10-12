"""Automated benchmark runner evaluating Tool Calling and Execution Success for Milestone E3."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel
import soundfile as sf

from src.agent.executor import ToolExecutor
from src.agent.schemas import EvidencePackage
from src.evaluation.llm_benchmark import load_benchmark_cases
from src.llm.client import OllamaClient


class ExecutionCaseResult(BaseModel):
    case_id: str
    question: str
    expected_tools: List[str]
    is_abstention: bool
    tools_dispatched: List[str]
    execution_status: str
    is_success: bool
    dsp_latency_ms: float
    total_pipeline_latency_ms: float
    error_or_warning: Optional[str] = None


class ToolExecutionBenchmarkSummary(BaseModel):
    total_cases: int
    successful_executions: int
    execution_success_rate: float
    average_dsp_latency_ms: float
    average_total_latency_ms: float
    results: List[ExecutionCaseResult]


def get_test_waveform(audio_type: str = "synthetic_sine_1000hz") -> tuple[np.ndarray, int]:
    """Generate the appropriate synthetic waveform for the benchmark case."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if audio_type == "synthetic_white_noise":
        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.25, int(sr * duration)).astype(np.float32)
        return np.clip(y, -0.9, 0.9), sr
    # Default 1000 Hz sine
    y = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    return y, sr


def run_execution_benchmark(
    cases_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> ToolExecutionBenchmarkSummary:
    """Run E3 benchmark evaluating full tool dispatch and execution success rate."""
    client = OllamaClient()
    executor = ToolExecutor(ollama_client=client)
    cases = load_benchmark_cases(cases_path)
    total_cases = len(cases)

    # Prepare ephemeral dataset directory fixture for dataset-level cases
    temp_dir = Path("temp_bench_dataset")
    temp_dir.mkdir(exist_ok=True)
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(temp_dir / "sample1.wav"), (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    sf.write(str(temp_dir / "sample2.wav"), (0.4 * np.sin(2 * np.pi * 880 * t)).astype(np.float32), sr)

    if verbose:
        print(f"\n=======================================================")
        print(f"Running AuralQ Milestone E3 Tool Execution Benchmark")
        print(f"Total Test Cases: {total_cases}")
        print(f"=======================================================\n")

    results: List[ExecutionCaseResult] = []

    for i, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"BC_{i:03d}")
        question = case.get("question", "")
        expected_tools = case.get("expected_tools", [])
        expected_abstain = case.get("is_abstention", False)

        if verbose:
            print(f"[{i}/{total_cases}] Case {case_id}: \"{question[:50]}...\"")

        t_start = time.perf_counter()

        # Step 1: Parse intent via Ollama
        llm_res = client.parse_intent(question)
        if not llm_res.is_valid or llm_res.parsed_intent is None:
            results.append(
                ExecutionCaseResult(
                    case_id=case_id,
                    question=question,
                    expected_tools=expected_tools,
                    is_abstention=expected_abstain,
                    tools_dispatched=[],
                    execution_status="LLM_PARSE_ERROR",
                    is_success=False,
                    dsp_latency_ms=0.0,
                    total_pipeline_latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    error_or_warning=llm_res.error_message,
                )
            )
            continue

        # Step 2: Prepare appropriate audio/dataset input
        if "dataset" in question.lower() or "folder" in question.lower() or "dataset_profiler" in expected_tools:
            waveform, sr = None, 22050
            dataset_path = str(temp_dir.resolve())
        else:
            waveform, sr = get_test_waveform("synthetic_white_noise" if "bright" in question.lower() or "hiss" in question.lower() else "synthetic_sine_1000hz")
            dataset_path = None

        # Step 3: Execute tool intent via ToolExecutor
        evidence: EvidencePackage = executor.execute_intent(
            intent_result=llm_res.parsed_intent,
            waveform=waveform,
            sr=sr,
            dataset_path=dataset_path,
            query=question,
        )

        total_lat = round((time.perf_counter() - t_start) * 1000, 2)

        # Successful if DSP executed (SUCCESS or PARTIAL) or if clean abstention was intended
        is_success = (evidence.status in ["SUCCESS", "PARTIAL"]) or (expected_abstain and evidence.status == "ABSTAIN")

        res_item = ExecutionCaseResult(
            case_id=case_id,
            question=question,
            expected_tools=expected_tools,
            is_abstention=expected_abstain,
            tools_dispatched=evidence.tools_called,
            execution_status=evidence.status,
            is_success=is_success,
            dsp_latency_ms=evidence.total_dsp_latency_ms,
            total_pipeline_latency_ms=total_lat,
            error_or_warning="; ".join(evidence.warnings) if evidence.warnings else None,
        )
        results.append(res_item)

        if verbose:
            status_str = "SUCCESS" if is_success else "FAILED"
            print(f"      -> Status: {evidence.status} | Dispatched: {evidence.tools_called} | DSP: {evidence.total_dsp_latency_ms:.1f}ms | Total: {total_lat:.1f}ms | {status_str}\n")

    # Cleanup temporary dataset fixture
    try:
        for f in temp_dir.iterdir():
            f.unlink()
        temp_dir.rmdir()
    except Exception:
        pass

    success_count = sum(1 for r in results if r.is_success)
    success_rate = round(success_count / total_cases, 4)
    avg_dsp = sum(r.dsp_latency_ms for r in results) / total_cases
    avg_total = sum(r.total_pipeline_latency_ms for r in results) / total_cases

    summary = ToolExecutionBenchmarkSummary(
        total_cases=total_cases,
        successful_executions=success_count,
        execution_success_rate=success_rate,
        average_dsp_latency_ms=round(avg_dsp, 2),
        average_total_latency_ms=round(avg_total, 2),
        results=results,
    )

    if verbose:
        print("\n-------------------------------------------------------")
        print("MILESTONE E3 TOOL EXECUTION BENCHMARK RESULTS")
        print("-------------------------------------------------------")
        print(f"Execution Success Rate: {summary.execution_success_rate * 100:.1f}% (Exit Target: > 90.0%)")
        print(f"Successful Cases: {summary.successful_executions} / {summary.total_cases}")
        print(f"Average DSP Execution Time: {summary.average_dsp_latency_ms:.1f} ms")
        print(f"Average Total Pipeline Latency: {summary.average_total_latency_ms:.1f} ms")
        print("-------------------------------------------------------\n")

    return summary


if __name__ == "__main__":
    run_execution_benchmark()

