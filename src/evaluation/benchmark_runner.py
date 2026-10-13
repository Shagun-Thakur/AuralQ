"""Comprehensive Benchmark Runner evaluating Baselines B1–B4 for Milestone E6."""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf

from src.agent.orchestrator import AuralQAgent
from src.audio.loader import load_audio_file
from src.evaluation.baselines import (
    run_baseline_b1,
    run_baseline_b2,
    run_baseline_b3,
    run_baseline_b4,
)
from src.evaluation.llm_benchmark import load_benchmark_cases
from src.evaluation.metrics import (
    BaselineCaseResult,
    BaselineSummary,
    BenchmarkComparisonReport,
    compute_baseline_summary,
)
from src.llm.client import OllamaClient


def generate_benchmark_waveform(case: Dict[str, Any]) -> tuple[np.ndarray, int, str]:
    """Generate appropriate synthetic audio fixture matching the question characteristics."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    q = case.get("question", "").lower()

    if "noise" in q or "hiss" in q or "dull" in q:
        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.25, int(sr * duration)).astype(np.float32)
        return np.clip(y, -0.9, 0.9), sr, "white_noise"

    # Default 1000 Hz pure sine wave
    y = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    return y, sr, "sine_1000hz"


def run_full_benchmark(
    cases_path: Optional[str | Path] = None,
    baselines_to_run: Optional[List[str]] = None,
    output_json_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> BenchmarkComparisonReport:
    """
    Execute all 4 comparative baselines (B1–B4) on the benchmark cases and generate
    a comprehensive statistical evaluation report.
    """
    cases = load_benchmark_cases(cases_path)
    total_cases = len(cases)
    target_baselines = baselines_to_run or ["B1", "B2", "B3", "B4"]

    client = OllamaClient()
    agent = AuralQAgent(client=client)

    # Set up temporary directories for benchmark fixtures
    temp_dir = Path("temp_bench_e6")
    temp_dir.mkdir(exist_ok=True)
    temp_dataset_dir = temp_dir / "dataset"
    temp_dataset_dir.mkdir(exist_ok=True)

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(temp_dataset_dir / "sample1.wav"), (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    sf.write(str(temp_dataset_dir / "sample2.wav"), (0.4 * np.sin(2 * np.pi * 880 * t)).astype(np.float32), sr)

    temp_audio_file = temp_dir / "test_clip.wav"

    results_b1: List[BaselineCaseResult] = []
    results_b2: List[BaselineCaseResult] = []
    results_b3: List[BaselineCaseResult] = []
    results_b4: List[BaselineCaseResult] = []

    if verbose:
        print("\n=======================================================")
        print("Running AuralQ Milestone E6 Comparative Benchmark (B1–B4)")
        print(f"Total Test Cases: {total_cases}")
        print(f"Baselines Evaluated: {', '.join(target_baselines)}")
        print("=======================================================\n")

    for i, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"BC_{i:03d}")
        question = case.get("question", "")
        expected_tools = case.get("expected_tools", [])
        is_dataset = "dataset" in question.lower() or "folder" in question.lower()

        if verbose:
            print(f"[{i}/{total_cases}] Evaluating Case {case_id}: \"{question[:50]}...\"")

        # Prepare audio inputs
        if is_dataset:
            waveform, sample_rate = None, sr
            dataset_path = str(temp_dataset_dir.resolve())
            audio_path = None
            metadata = None
        else:
            waveform, sample_rate, _ = generate_benchmark_waveform(case)
            sf.write(str(temp_audio_file), waveform, sample_rate)
            dataset_path = None
            audio_path = str(temp_audio_file.resolve())
            load_res = load_audio_file(temp_audio_file)
            metadata = load_res.metadata

        # --- Baseline B1: Oracle / DSP Reference ---
        if "B1" in target_baselines:
            res_b1 = run_baseline_b1(
                case=case,
                waveform=waveform,
                sr=sample_rate,
                dataset_path=dataset_path,
                audio_metadata=metadata,
            )
            results_b1.append(res_b1)
            if verbose:
                print(f"     B1 (Oracle): Tools={res_b1.tools_executed} | Grounding={res_b1.grounding_score*100:.0f}% | DSP={res_b1.dsp_latency_ms:.1f}ms")

        # --- Baseline B2: LLM-Only (No DSP Tools) ---
        if "B2" in target_baselines:
            res_b2 = run_baseline_b2(
                case=case,
                b1_result=res_b1 if "B1" in target_baselines else None,
                client=client,
                audio_metadata=metadata,
            )
            results_b2.append(res_b2)
            if verbose:
                print(f"     B2 (LLM-Only): Hallucinations={len(res_b2.hallucinated_tokens)} | Grounding={res_b2.grounding_score*100:.0f}% | Total={res_b2.total_latency_ms:.1f}ms")

        # --- Baseline B3: Fixed Pipeline (All Tools) ---
        if "B3" in target_baselines:
            res_b3 = run_baseline_b3(
                case=case,
                waveform=waveform,
                sr=sample_rate,
                dataset_path=dataset_path,
                client=client,
                audio_metadata=metadata,
            )
            results_b3.append(res_b3)
            if verbose:
                print(f"     B3 (Fixed Pipeline): Tools={len(res_b3.tools_executed)} | DSP={res_b3.dsp_latency_ms:.1f}ms | Total={res_b3.total_latency_ms:.1f}ms")

        # --- Baseline B4: AuralQ Proposed Full Pipeline ---
        if "B4" in target_baselines:
            agent.session_state.clear_active_inputs()
            res_b4 = run_baseline_b4(
                case=case,
                waveform=waveform,
                sr=sample_rate,
                dataset_path=dataset_path,
                agent=agent,
                audio_path=audio_path,
            )
            results_b4.append(res_b4)
            if verbose:
                print(f"     B4 (AuralQ): Tools={res_b4.tools_executed} | Grounding={res_b4.grounding_score*100:.0f}% | Repaired={res_b4.was_repaired} | Total={res_b4.total_latency_ms:.1f}ms")

        if verbose:
            print()

    # Clean up temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Compute baseline summaries
    summaries: Dict[str, BaselineSummary] = {}
    if "B1" in target_baselines:
        summaries["B1"] = compute_baseline_summary("B1", "Oracle / DSP Ground Truth", results_b1)
    if "B2" in target_baselines:
        summaries["B2"] = compute_baseline_summary("B2", "LLM-Only (No DSP)", results_b2)
    if "B3" in target_baselines:
        summaries["B3"] = compute_baseline_summary("B3", "Fixed Pipeline (All Tools)", results_b3)
    if "B4" in target_baselines:
        summaries["B4"] = compute_baseline_summary("B4", "AuralQ Proposed Full Pipeline", results_b4)

    report = BenchmarkComparisonReport(baselines=summaries)

    if verbose:
        print("\n=======================================================")
        print("MILESTONE E6 COMPARATIVE BENCHMARK REPORT (B1–B4)")
        print("=======================================================\n")
        print(report.to_markdown_table())
        print("\n=======================================================\n")

    # Save to JSON
    out_path = Path(output_json_path or "data/benchmark/benchmark_results_b1_b4.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    return report


if __name__ == "__main__":
    run_full_benchmark(verbose=True)

