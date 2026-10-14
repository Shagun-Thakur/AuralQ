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
7. PSYCHOACOUSTIC REASONING:
   - Frequencies < 500 Hz indicate low-frequency bass concentration (dark / dull).
   - Frequencies between 500 Hz and 2000 Hz form the auditory midrange. A concentrated pure tone around 1 kHz (indicated by narrow standard deviation or low bandwidth) sits right in the human ear's high-sensitivity resonance zone and is perceived as sharp, clear, or piercing rather than dull.
   - Frequencies > 2500 Hz indicate upper-mid/treble presence or crispness (bright / sharp).
   - Characterize whether the sound is a pure tone or broadband based on the measured standard deviation, bandwidth, and qualitative anchor.
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

    def _format_dataset_profile_report(self, evidence: EvidencePackage) -> str:
        """Format dataset audit evidence into a clean, structured executive report."""
        metrics = evidence.consolidated_metrics
        path = metrics.get("dataset_profiler.dataset_path", "Dataset")
        total_files = metrics.get("dataset_profiler.total_files", 0)
        total_dur_hours = metrics.get("dataset_profiler.total_duration_hours", 0.0)
        total_dur_sec = round(total_dur_hours * 3600, 2)
        sr_dist = metrics.get("dataset_profiler.sample_rate_distribution", {})
        ch_dist = metrics.get("dataset_profiler.channel_distribution", {})
        silent_count = metrics.get("dataset_profiler.silent_files_count", 0)
        clipped_count = metrics.get("dataset_profiler.clipped_files_count", 0)
        corrupted_count = metrics.get("dataset_profiler.corrupted_files_count", 0)
        recs = metrics.get("dataset_profiler.recommendations", [])
        centroid = metrics.get("dataset_profiler.spectral_centroid_mean_hz")
        flatness = metrics.get("dataset_profiler.spectral_flatness_mean")

        sr_str = ", ".join(f"{sr} Hz ({c} files)" for sr, c in sorted(sr_dist.items())) if sr_dist else "Uniform"
        ch_str = ", ".join(f"{'Mono' if ch == 1 else ('Stereo' if ch == 2 else f'{ch}-channel')} ({c} files)" for ch, c in sorted(ch_dist.items())) if ch_dist else "Uniform"

        lines = [
            f"### 📊 Audio Dataset Audit Summary: `{path}`",
            f"- **Dataset Volume**: {total_files} audio file(s) ({total_dur_hours:.3f} hours / {total_dur_sec:.1f}s)",
            f"- **Sample Rate Distribution**: {sr_str}",
            f"- **Channel Layout**: {ch_str}",
            f"- **Acoustic Hygiene Anomalies**: {silent_count} silent file(s), {clipped_count} clipped file(s), {corrupted_count} corrupted file(s).",
        ]

        if centroid is not None and flatness is not None:
            lines.append(f"- **Mean Spectral Centroid**: {centroid:.2f} Hz | **Mean Spectral Flatness**: {flatness:.4f}")

        if evidence.consolidated_anchors:
            anchors_clean = [f"**{k.replace('dataset_profiler.', '').replace('_', ' ').title()}**: `{v}`" for k, v in evidence.consolidated_anchors.items()]
            lines.append(f"- **Acoustic Assessment**: {', '.join(anchors_clean)}")

        if recs:
            lines.append("\n### 🛠️ Recommended Preprocessing Pipeline:")
            for idx, r in enumerate(recs, 1):
                prio = r.get("priority", "RECOMMENDED")
                action = r.get("action", "ACTION")
                reason = r.get("reason", "")
                params = r.get("parameters", {})
                param_strs = [f"`{k}={v}`" for k, v in params.items()]
                param_formatted = ", ".join(param_strs) if param_strs else "default"
                lines.append(f"{idx}. **[{prio}] {action}**")
                lines.append(f"   - **Reason**: {reason}")
                lines.append(f"   - **Parameters**: {param_formatted}")
        else:
            lines.append("\n✅ **No preprocessing required**: All audio files exhibit consistent acoustic hygiene.")

        if evidence.warnings:
            lines.append(f"\n*Warnings: {'; '.join(evidence.warnings)}*")

        return "\n".join(lines)

    def _deterministic_fallback_synthesis(self, evidence: EvidencePackage) -> str:
        """Rule-based template synthesis used as a resilient fallback if LLM generation fails."""
        lines = []

        if evidence.status == "TOOL_FAILURE":
            return f"Acoustic measurement failed for this query. Warnings: {'; '.join(evidence.warnings)}"

        # Check if dataset profiler was executed
        if any(k.startswith("dataset_profiler.") for k in evidence.consolidated_metrics):
            return self._format_dataset_profile_report(evidence)

        # Report perceptual anchors
        if evidence.consolidated_anchors:
            anchor_strs = [f"{k}: '{v}'" for k, v in evidence.consolidated_anchors.items()]
            lines.append(f"Acoustic assessment indicates {', '.join(anchor_strs)}.")

        # Report measured metrics cleanly (skipping raw dict dumps)
        if evidence.consolidated_metrics:
            metric_strs = []
            for k, v in evidence.consolidated_metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    metric_strs.append(f"{k} = {v}")
                elif isinstance(v, dict):
                    inner = ", ".join(f"{subk}={subv}" for subk, subv in v.items() if isinstance(subv, (int, float, str, bool)))
                    if inner:
                        metric_strs.append(f"{k} = {{{inner}}}")
            if metric_strs:
                lines.append(f"Measured values: {', '.join(metric_strs)}.")

        # Disclose any warnings
        if evidence.warnings:
            lines.append(f"Note: {'; '.join(evidence.warnings)}.")

        if not lines:
            lines.append("Acoustic analysis completed with no anomalous features detected.")

        return " ".join(lines)


