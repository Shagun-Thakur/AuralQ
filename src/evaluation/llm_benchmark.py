"""Automated benchmark runner evaluating LLM tool selection and schema compliance for Milestone E2."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.llm.client import OllamaClient
from src.llm.schemas import OllamaResponse


class CaseEvaluationResult(BaseModel):
    case_id: str
    question: str
    expected_tools: List[str]
    is_expected_abstention: bool
    is_schema_valid: bool
    actual_tools: List[str]
    actual_abstention: bool
    tool_match_score: float  # Jaccard similarity or precision
    latency_ms: float
    retry_count: int
    status: str
    error_message: Optional[str] = None


class BenchmarkSummary(BaseModel):
    model_name: str
    total_cases: int
    schema_valid_count: int
    schema_valid_rate: float
    exact_tool_match_count: int
    tool_accuracy_rate: float
    abstention_accuracy_rate: float
    average_latency_ms: float
    p95_latency_ms: float
    results: List[CaseEvaluationResult]


def load_benchmark_cases(cases_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """Load benchmark cases from JSON file."""
    path = Path(cases_path) if cases_path else Path(__file__).resolve().parent.parent.parent / "data" / "benchmark" / "cases.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_case(
    client: OllamaClient,
    case: Dict[str, Any],
) -> CaseEvaluationResult:
    """Evaluate a single benchmark case through the Ollama client."""
    case_id = case.get("case_id", "UNKNOWN")
    question = case.get("question", "")
    expected_tools = case.get("expected_tools", [])
    expected_abstain = case.get("is_abstention", False)

    res: OllamaResponse = client.parse_intent(question)

    if not res.is_valid or res.parsed_intent is None:
        return CaseEvaluationResult(
            case_id=case_id,
            question=question,
            expected_tools=expected_tools,
            is_expected_abstention=expected_abstain,
            is_schema_valid=False,
            actual_tools=[],
            actual_abstention=False,
            tool_match_score=0.0,
            latency_ms=res.metadata.total_duration_ms if res.metadata else 0.0,
            retry_count=res.retry_count,
            status=res.status,
            error_message=res.error_message,
        )

    parsed = res.parsed_intent
    actual_tools = [t.tool_name for t in parsed.selected_tools]
    actual_abstain = parsed.is_abstention

    # Calculate tool match score
    if expected_abstain:
        match_score = 1.0 if actual_abstain else 0.0
    else:
        exp_set = set(expected_tools)
        act_set = set(actual_tools)
        if not exp_set and not act_set:
            match_score = 1.0
        elif not exp_set or not act_set:
            match_score = 0.0
        else:
            intersection = len(exp_set.intersection(act_set))
            union = len(exp_set.union(act_set))
            match_score = round(intersection / union, 2)

    return CaseEvaluationResult(
        case_id=case_id,
        question=question,
        expected_tools=expected_tools,
        is_expected_abstention=expected_abstain,
        is_schema_valid=True,
        actual_tools=actual_tools,
        actual_abstention=actual_abstain,
        tool_match_score=match_score,
        latency_ms=res.metadata.total_duration_ms if res.metadata else 0.0,
        retry_count=res.retry_count,
        status=res.status,
    )


def run_benchmark(
    model_name: Optional[str] = None,
    cases_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> BenchmarkSummary:
    """Run the 20-case benchmark and compute Milestone E2 exit metrics."""
    client = OllamaClient(model_name=model_name)
    cases = load_benchmark_cases(cases_path)
    total_cases = len(cases)

    if verbose:
        print(f"\n=======================================================")
        print(f"Running AuralQ Milestone E2 Benchmark: {client.model_name}")
        print(f"Total Test Cases: {total_cases}")
        print(f"=======================================================\n")

    eval_results: List[CaseEvaluationResult] = []
    latencies: List[float] = []

    for i, case in enumerate(cases, 1):
        if verbose:
            print(f"[{i}/{total_cases}] Evaluating {case['case_id']}: \"{case['question'][:50]}...\"")
        res = evaluate_case(client, case)
        eval_results.append(res)
        latencies.append(res.latency_ms)

        if verbose:
            match_str = "MATCH" if res.tool_match_score == 1.0 else f"PARTIAL ({res.tool_match_score})" if res.tool_match_score > 0 else "MISMATCH"
            print(f"      -> Valid: {res.is_schema_valid} | Tools: {res.actual_tools} (Exp: {res.expected_tools}) | {match_str} | {res.latency_ms:.1f}ms\n")

    valid_count = sum(1 for r in eval_results if r.is_schema_valid)
    exact_tool_matches = sum(1 for r in eval_results if r.tool_match_score == 1.0)
    
    abstain_expected = [r for r in eval_results if r.is_expected_abstention]
    abstain_correct = sum(1 for r in abstain_expected if r.actual_abstention)
    abstain_rate = (abstain_correct / len(abstain_expected)) if abstain_expected else 1.0

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0.0

    summary = BenchmarkSummary(
        model_name=client.model_name,
        total_cases=total_cases,
        schema_valid_count=valid_count,
        schema_valid_rate=round(valid_count / total_cases, 4),
        exact_tool_match_count=exact_tool_matches,
        tool_accuracy_rate=round(exact_tool_matches / total_cases, 4),
        abstention_accuracy_rate=round(abstain_rate, 4),
        average_latency_ms=round(avg_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        results=eval_results,
    )

    if verbose:
        print("\n-------------------------------------------------------")
        print("MILESTONE E2 BENCHMARK RESULTS")
        print("-------------------------------------------------------")
        print(f"Model: {summary.model_name}")
        print(f"Schema-Valid Rate: {summary.schema_valid_rate * 100:.1f}% (Exit Target: >= 80.0%)")
        print(f"Exact Tool Match Rate: {summary.tool_accuracy_rate * 100:.1f}%")
        print(f"Abstention Accuracy: {summary.abstention_accuracy_rate * 100:.1f}%")
        print(f"Average Latency: {summary.average_latency_ms:.1f} ms")
        print(f"P95 Latency: {summary.p95_latency_ms:.1f} ms")
        print("-------------------------------------------------------\n")

    return summary


if __name__ == "__main__":
    run_benchmark()

