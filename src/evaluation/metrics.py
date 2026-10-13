"""Statistical metrics, Pydantic schemas, and comparative reporting for AuralQ Baselines B1–B4."""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BaselineCaseResult(BaseModel):
    """Evaluation result for an individual benchmark test case under a specific baseline."""
    baseline_id: str = Field(description="Identifier of the baseline ('B1', 'B2', 'B3', 'B4').")
    case_id: str = Field(description="Benchmark case ID (e.g. 'BC_001').")
    question: str = Field(description="User prompt evaluated.")
    tools_executed: List[str] = Field(default_factory=list, description="List of DSP tools actually computed.")
    tool_count: int = Field(default=0, description="Number of DSP tools executed.")
    is_abstention: bool = Field(default=False, description="Whether this baseline treated the query as an abstention.")
    abstention_correct: bool = Field(default=True, description="Whether the abstention status matched the ground truth.")
    response_text: str = Field(description="Natural-language response produced.")
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Citation grounding score (0.0 to 1.0).")
    is_fully_grounded: bool = Field(default=True, description="True if zero hallucinated numbers were detected.")
    hallucinated_tokens: List[str] = Field(default_factory=list, description="List of ungrounded numeric tokens detected.")
    was_repaired: bool = Field(default=False, description="True if response was repaired via deterministic fallback.")
    dsp_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class BaselineSummary(BaseModel):
    """Aggregated statistical summary for a single baseline across all benchmark cases."""
    baseline_id: str
    baseline_name: str
    total_cases: int
    avg_tools_per_query: float
    abstention_accuracy: float
    citation_grounding_rate: float
    numeric_hallucination_rate: float  # % of cases with >= 1 ungrounded number
    avg_dsp_latency_ms: float
    avg_llm_latency_ms: float
    avg_total_latency_ms: float
    repair_rate: float
    case_results: List[BaselineCaseResult] = Field(default_factory=list)


def compute_baseline_summary(
    baseline_id: str,
    baseline_name: str,
    results: List[BaselineCaseResult],
) -> BaselineSummary:
    """Compute aggregate evaluation metrics across benchmark cases for a baseline."""
    n = len(results)
    if n == 0:
        return BaselineSummary(
            baseline_id=baseline_id,
            baseline_name=baseline_name,
            total_cases=0,
            avg_tools_per_query=0.0,
            abstention_accuracy=0.0,
            citation_grounding_rate=0.0,
            numeric_hallucination_rate=0.0,
            avg_dsp_latency_ms=0.0,
            avg_llm_latency_ms=0.0,
            avg_total_latency_ms=0.0,
            repair_rate=0.0,
            case_results=[],
        )

    avg_tools = round(sum(r.tool_count for r in results) / n, 2)
    abstain_correct = sum(1 for r in results if r.abstention_correct)
    abstain_acc = round((abstain_correct / n) * 100.0, 1)

    avg_grounding = round((sum(r.grounding_score for r in results) / n) * 100.0, 1)
    hallucinated_cases = sum(1 for r in results if not r.is_fully_grounded)
    hallucination_rate = round((hallucinated_cases / n) * 100.0, 1)

    avg_dsp_lat = round(sum(r.dsp_latency_ms for r in results) / n, 2)
    avg_llm_lat = round(sum(r.llm_latency_ms for r in results) / n, 2)
    avg_tot_lat = round(sum(r.total_latency_ms for r in results) / n, 2)
    repaired_count = sum(1 for r in results if r.was_repaired)
    repair_rate = round((repaired_count / n) * 100.0, 1)

    return BaselineSummary(
        baseline_id=baseline_id,
        baseline_name=baseline_name,
        total_cases=n,
        avg_tools_per_query=avg_tools,
        abstention_accuracy=abstain_acc,
        citation_grounding_rate=avg_grounding,
        numeric_hallucination_rate=hallucination_rate,
        avg_dsp_latency_ms=avg_dsp_lat,
        avg_llm_latency_ms=avg_llm_lat,
        avg_total_latency_ms=avg_tot_lat,
        repair_rate=repair_rate,
        case_results=results,
    )


class BenchmarkComparisonReport(BaseModel):
    """Comparative evaluation report across all 4 baselines (B1–B4)."""
    timestamp: float = Field(default_factory=time.time)
    model_name: str = "llama3.2:3b-instruct-q4_K_M"
    hardware_profile: str = "NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), AMD Ryzen 5 5600H, 16 GB RAM"
    baselines: Dict[str, BaselineSummary]

    def to_markdown_table(self) -> str:
        """Format the comparative results into a clean markdown table for reports."""
        lines = [
            "| Metric | B1: Oracle / DSP Ground Truth | B2: LLM-Only (No DSP) | B3: Fixed Pipeline (All Tools) | B4: AuralQ Proposed |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        b1 = self.baselines.get("B1")
        b2 = self.baselines.get("B2")
        b3 = self.baselines.get("B3")
        b4 = self.baselines.get("B4")

        def val(b: Optional[BaselineSummary], attr: str, suffix: str = "") -> str:
            if not b:
                return "N/A"
            v = getattr(b, attr, "N/A")
            return f"{v}{suffix}"

        lines.append(f"| **Citation Grounding Rate** | {val(b1, 'citation_grounding_rate', '%')} | {val(b2, 'citation_grounding_rate', '%')} | {val(b3, 'citation_grounding_rate', '%')} | **{val(b4, 'citation_grounding_rate', '%')}** |")
        lines.append(f"| **Numeric Hallucination Rate** | {val(b1, 'numeric_hallucination_rate', '%')} | {val(b2, 'numeric_hallucination_rate', '%')} | {val(b3, 'numeric_hallucination_rate', '%')} | **{val(b4, 'numeric_hallucination_rate', '%')}** |")
        lines.append(f"| **Abstention Accuracy** | {val(b1, 'abstention_accuracy', '%')} | {val(b2, 'abstention_accuracy', '%')} | {val(b3, 'abstention_accuracy', '%')} | **{val(b4, 'abstention_accuracy', '%')}** |")
        lines.append(f"| **Avg Tools Executed / Query** | {val(b1, 'avg_tools_per_query')} | {val(b2, 'avg_tools_per_query')} | {val(b3, 'avg_tools_per_query')} | **{val(b4, 'avg_tools_per_query')}** |")
        lines.append(f"| **Avg DSP Compute Time** | {val(b1, 'avg_dsp_latency_ms', ' ms')} | {val(b2, 'avg_dsp_latency_ms', ' ms')} | {val(b3, 'avg_dsp_latency_ms', ' ms')} | **{val(b4, 'avg_dsp_latency_ms', ' ms')}** |")
        lines.append(f"| **Avg End-to-End Latency** | {val(b1, 'avg_total_latency_ms', ' ms')} | {val(b2, 'avg_total_latency_ms', ' ms')} | {val(b3, 'avg_total_latency_ms', ' ms')} | **{val(b4, 'avg_total_latency_ms', ' ms')}** |")
        lines.append(f"| **Deterministic Fallback Rate** | {val(b1, 'repair_rate', '%')} | {val(b2, 'repair_rate', '%')} | {val(b3, 'repair_rate', '%')} | **{val(b4, 'repair_rate', '%')}** |")

        return "\n".join(lines)

