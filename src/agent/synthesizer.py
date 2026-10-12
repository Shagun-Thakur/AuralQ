"""Layer 4: Natural-Language Response Synthesizer grounded strictly in deterministic DSP evidence."""

import time
from typing import Any, Dict, List, Optional, Tuple

from src.agent.schemas import EvidencePackage, TurnRecord
from src.llm.client import OllamaClient


SYSTEM_SYNTHESIS_PROMPT = """You are AuralQ's Grounded Audio Intelligence Explainer.
Your task is to provide an accurate, clear, and concise (under 150 words) natural language answer to the user's question, strictly grounded in the provided acoustic evidence.

CRITICAL GROUNDING RULES:
1. STRICT NUMERIC GROUNDING: Every number, frequency (Hz), percentage, or ratio you cite MUST exist in the <evidence> block. Never extrapolate, hallucinate, or calculate acoustic metrics yourself.
2. PERCEPTUAL ANCHORS: Explicitly reference the qualitative perceptual anchor(s) provided in the evidence (e.g. 'bright', 'dull', 'tonal', 'noise-like', 'consistent amplitude', 'fluctuating').
3. CITATION OF MEASUREMENTS: When explaining acoustic properties, cite the exact measured metric with its unit (e.g., "spectral centroid of 3421.5 Hz", "mean RMS energy of 0.082").
4. WARNING DISCLOSURES: If the evidence contains INGESTION_WARNINGS or DSP_WARNINGS (such as clipping or silence), you MUST explicitly mention them as cautionary notes.
5. ABSTENTION BEHAVIOR: If STATUS is ABSTAIN, politely explain the abstention reason and clarify that AuralQ specializes strictly in deterministic acoustic signal processing and dataset profiling.
6. CONCISENESS & TONE: Keep the response direct, scientific, and concise (under 150 words). Avoid generic introductory fluff (e.g. "Sure! Here is what I found:").
"""


class ResponseSynthesizer:
    """
    Synthesizes natural-language user responses strictly grounded in the DSP evidence package.
    Includes a deterministic template fallback when local LLM inference is degraded or unavailable.
    """

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()

    def build_synthesis_prompt(
        self,
        query: str,
        evidence: EvidencePackage,
        history_context: str = "",
    ) -> str:
        """Construct the prompt combining query, conversation history, and evidence package."""
        parts = []
        if history_context:
            parts.append(history_context)

        parts.append(f"USER QUERY: {query.strip()}")
        parts.append("\n" + evidence.format_evidence_for_llm())
        parts.append("\nAnswer the query concisely, citing the exact measured metrics and perceptual anchors:")
        return "\n".join(parts)

    def synthesize(
        self,
        query: str,
        evidence: EvidencePackage,
        history_context: str = "",
    ) -> Tuple[str, float]:
        """
        Synthesize natural-language answer.
        Returns (response_text, synthesis_latency_ms).
        """
        # Fast-path for explicit abstention
        if evidence.status == "ABSTAIN":
            reason = evidence.abstention_reason or "Question is outside acoustic signal processing scope."
            text = f"I cannot answer this query because it is outside the scope of acoustic signal processing. {reason}"
            return text, 0.0

        t_start = time.perf_counter()
        prompt = self.build_synthesis_prompt(query, evidence, history_context)

        try:
            res_json = self.client.generate_text(
                prompt=prompt,
                system=SYSTEM_SYNTHESIS_PROMPT,
            )
            synthesis_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
            raw_text = res_json.get("response", "").strip()
            if raw_text:
                return raw_text, synthesis_time_ms
        except Exception:
            # Fall back to deterministic template synthesis on any LLM connection issue
            pass

        # Deterministic fallback synthesis
        synthesis_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
        fallback_text = self._deterministic_fallback_synthesis(evidence)
        return fallback_text, synthesis_time_ms

    def _deterministic_fallback_synthesis(self, evidence: EvidencePackage) -> str:
        """Rule-based template synthesis used as a resilient fallback if LLM generation fails."""
        lines = []

        if evidence.status == "TOOL_FAILURE":
            return f"Acoustic measurement failed for this query. Warnings: {'; '.join(evidence.warnings)}"

        # Report perceptual anchors
        if evidence.consolidated_anchors:
            anchor_strs = [f"{k}: '{v}'" for k, v in evidence.consolidated_anchors.items()]
            lines.append(f"Acoustic assessment indicates {', '.join(anchor_strs)}.")

        # Report measured metrics
        if evidence.consolidated_metrics:
            metric_strs = [f"{k} = {v}" for k, v in evidence.consolidated_metrics.items()]
            lines.append(f"Measured values: {', '.join(metric_strs)}.")

        # Disclose any warnings
        if evidence.warnings:
            lines.append(f"Note: {'; '.join(evidence.warnings)}.")

        if not lines:
            lines.append("Acoustic analysis completed with no anomalous features detected.")

        return " ".join(lines)

