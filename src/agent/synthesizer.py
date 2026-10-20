"""Layer 4: Natural-Language Response Synthesizer grounded strictly in deterministic DSP evidence."""

import time
from typing import Any, Dict, List, Optional, Tuple

from src.agent.schemas import EvidencePackage, TurnRecord
from src.llm.client import OllamaClient


SYSTEM_SYNTHESIS_PROMPT = """You are AuralQ's Grounded Audio Intelligence Explainer.
Your task is to provide an accurate, clear, and concise (under 160 words) natural language answer to the user's question, strictly grounded in the provided acoustic evidence.

CRITICAL GROUNDING RULES:
1. STRICT NUMERIC GROUNDING: Every measured file number, frequency (Hz), percentage, or ratio you cite MUST exist in the <evidence> block. Never extrapolate, hallucinate, or calculate measured file metrics yourself.
2. PERCEPTUAL ANCHORS: Explicitly reference the qualitative perceptual anchor(s) provided in the evidence (e.g. 'bright', 'dull', 'tonal', 'noise-like', 'consistent amplitude', 'fluctuating').
3. CITATION OF MEASUREMENTS: When explaining acoustic properties, cite the exact measured metric with its unit (e.g., "spectral centroid of 5319.11 Hz", "mean RMS energy of 0.3043 (-10.33 dBFS)").
4. WARNING DISCLOSURES: If the evidence contains INGESTION_WARNINGS or DSP_WARNINGS (such as clipping or silence), you MUST explicitly mention them as cautionary notes.
5. ABSTENTION BEHAVIOR: If STATUS is ABSTAIN, politely explain that the query falls outside audio physics and clarify that AuralQ specializes in acoustic signal processing and dataset profiling.
6. CONCISENESS & TONE: Keep the response direct, scientific, professional, and concise. Avoid generic introductory fluff (e.g. "Sure! Here is what I found:").
7. PSYCHOACOUSTIC REASONING:
   - Frequencies < 500 Hz indicate low-frequency bass concentration (dark / dull).
   - Frequencies between 500 Hz and 2000 Hz form the auditory midrange. A concentrated pure tone around 1 kHz (indicated by narrow standard deviation or low bandwidth) sits right in the human ear's high-sensitivity resonance zone and is perceived as sharp, clear, or piercing rather than dull.
   - Frequencies > 2500 Hz indicate upper-mid/treble presence or crispness (bright / sharp).
   - Characterize whether the sound is a pure tone or broadband based on the measured standard deviation, bandwidth, and qualitative anchor.
8. APPLIED USE-CASE EVALUATION (VLOGS, PODCASTS, INTROS, VOICEOVER):
   - When asked if the audio is suitable for an applied use-case (e.g. vlog background music, podcast backdrop, voiceover bed):
     * Directly assess suitability based on the measured metrics.
     * Dialogue Masking: Speech sits between -14 and -18 dBFS. Background audio must sit 12-18 dB lower (around -24 to -28 dBFS). If measured RMS is high (e.g. > -20 dBFS), warn that it will overpower dialogue.
     * Frequency Clashing: Speech intelligibility resides in 300 Hz-4 kHz, and sibilance in 4-7 kHz. A high spectral centroid (> 4 kHz) will fight voice consonants.
     * Provide concrete mixing guidance (e.g. lower volume by 12-15 dB, apply gentle high-shelf reduction).
9. REMEDIATION & PREPROCESSING STEPS (NOISE REDUCTION, CLEANING):
   - When asked what preprocessing or cleaning steps to take:
     * High-Pass Filter: Cut sub-bass rumble below 80 Hz with a 12 dB/oct slope.
     * Noise Gate / Expander: Set a threshold ~6 dB above the noise floor to silence pauses between speech/notes.
     * Spectral Denoising: Recommend a light 3 to 6 dB reduction to eliminate hiss without watery artifacts.
     * High-Shelf / Taming: If treble is sharp (> 5 kHz), suggest a subtle high-shelf cut above 6 kHz.
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

        raw_text = ""
        try:
            res_json = self.client.generate_text(
                prompt=prompt,
                system=SYSTEM_SYNTHESIS_PROMPT,
            )
            synthesis_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
            raw_text = res_json.get("response", "").strip()
            if raw_text and not raw_text.startswith("Cloud API error"):
                return raw_text, synthesis_time_ms
        except Exception:
            # Fall back to deterministic template synthesis on any LLM connection issue
            pass

        # Deterministic fallback synthesis
        synthesis_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
        fallback_text = self._deterministic_fallback_synthesis(evidence)

        # Disclose quota or service warning cleanly without raw numeric digits (so citation validator is not affected)
        if any(tok in raw_text for tok in ("429", "Too Many Requests", "RESOURCE_EXHAUSTED")):
            fallback_text += (
                "\n\n*⚠️ Note: Cloud LLM quota limit reached. "
                "This response was deterministically synthesized from verified DSP measurements. "
                "Supply your own key in the sidebar or run locally via `run_auralq.bat` for unlimited AI explanations.*"
            )
        elif any(tok in raw_text for tok in ("503", "Service Unavailable", "502", "504", "overloaded")):
            fallback_text += (
                "\n\n*⚠️ Note: Cloud AI servers are temporarily experiencing high traffic. "
                "This response was deterministically synthesized from verified DSP measurements. "
                "Run locally via `run_auralq.bat` for guaranteed zero-downtime execution.*"
            )

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

        # Report perceptual anchors cleanly without duplicate key/value dumps
        if evidence.consolidated_anchors:
            grounded_terms = [v for k, v in evidence.consolidated_anchors.items() if k.endswith(".grounded_term")]
            if grounded_terms:
                unique_terms = list(dict.fromkeys(grounded_terms))
                lines.append(f"Acoustic assessment indicates the audio is characterized as **{', '.join(unique_terms)}**.")
            else:
                unique_vals = list(dict.fromkeys(evidence.consolidated_anchors.values()))
                lines.append(f"Acoustic assessment indicates the audio is characterized as **{', '.join(unique_vals)}**.")

        # Report measured metrics cleanly with proper units
        metrics = evidence.consolidated_metrics
        if metrics:
            metric_descriptions = []
            handled_keys = set()

            # 1. Spectral Centroid
            if any(k.startswith("spectral_centroid.") for k in metrics):
                mean_hz = metrics.get("spectral_centroid.mean_hz") or metrics.get("spectral_centroid.mean_centroid_hz")
                med_hz = metrics.get("spectral_centroid.median_hz") or metrics.get("spectral_centroid.median_centroid_hz")
                std_hz = metrics.get("spectral_centroid.std_hz") or metrics.get("spectral_centroid.std_centroid_hz")
                handled_keys.update({"spectral_centroid.mean_hz", "spectral_centroid.mean_centroid_hz",
                                     "spectral_centroid.median_hz", "spectral_centroid.median_centroid_hz",
                                     "spectral_centroid.std_hz", "spectral_centroid.std_centroid_hz"})
                if mean_hz is not None:
                    s = f"mean spectral centroid is {mean_hz} Hz"
                    sub = []
                    if med_hz is not None:
                        sub.append(f"median: {med_hz} Hz")
                    if std_hz is not None:
                        sub.append(f"std: {std_hz} Hz")
                    if sub:
                        s += f" ({', '.join(sub)})"
                    metric_descriptions.append(s)

            # 2. RMS Energy
            if any(k.startswith("rms_energy.") for k in metrics):
                mean_rms = metrics.get("rms_energy.mean_rms")
                dbfs = metrics.get("rms_energy.mean_rms_dbfs")
                max_rms = metrics.get("rms_energy.max_rms")
                min_rms = metrics.get("rms_energy.min_rms")
                handled_keys.update({"rms_energy.mean_rms", "rms_energy.mean_rms_dbfs", "rms_energy.max_rms", "rms_energy.min_rms"})
                if mean_rms is not None:
                    s = f"mean RMS energy is {mean_rms}"
                    if dbfs is not None:
                        s += f" ({dbfs} dBFS)"
                    sub = []
                    if max_rms is not None:
                        sub.append(f"peak: {max_rms}")
                    if min_rms is not None:
                        sub.append(f"min: {min_rms}")
                    if sub:
                        s += f" (range: {', '.join(sub)})"
                    metric_descriptions.append(s)

            # 3. Pitch YIN
            if any(k.startswith("pitch_yin.") for k in metrics):
                mean_f0 = metrics.get("pitch_yin.mean_f0_hz")
                voiced = metrics.get("pitch_yin.voiced_fraction")
                handled_keys.update({"pitch_yin.mean_f0_hz", "pitch_yin.voiced_fraction"})
                if mean_f0 is not None:
                    s = f"fundamental frequency (F0) is {mean_f0} Hz"
                    if voiced is not None:
                        s += f" (voiced fraction: {voiced})"
                    metric_descriptions.append(s)

            # 4. Spectral Flatness
            if any(k.startswith("spectral_flatness.") for k in metrics):
                flatness = metrics.get("spectral_flatness.mean_flatness")
                tonality = metrics.get("spectral_flatness.tonality_coefficient")
                handled_keys.update({"spectral_flatness.mean_flatness", "spectral_flatness.tonality_coefficient"})
                if flatness is not None:
                    s = f"spectral flatness is {flatness}"
                    if tonality is not None:
                        s += f" (tonality: {tonality})"
                    metric_descriptions.append(s)

            # 5. Zero Crossing Rate
            if any(k.startswith("zero_crossing_rate.") for k in metrics):
                mean_zcr = metrics.get("zero_crossing_rate.mean_zcr")
                handled_keys.add("zero_crossing_rate.mean_zcr")
                if mean_zcr is not None:
                    metric_descriptions.append(f"zero-crossing rate is {mean_zcr}")

            # 6. SNR Estimate
            if any(k.startswith("snr_estimate.") for k in metrics):
                snr = metrics.get("snr_estimate.estimated_snr_db")
                handled_keys.add("snr_estimate.estimated_snr_db")
                if snr is not None:
                    metric_descriptions.append(f"estimated SNR is {snr} dB")

            # 7. Spectral Rolloff
            if any(k.startswith("spectral_rolloff.") for k in metrics):
                rolloff = metrics.get("spectral_rolloff.mean_rolloff_hz")
                handled_keys.add("spectral_rolloff.mean_rolloff_hz")
                if rolloff is not None:
                    metric_descriptions.append(f"spectral roll-off is {rolloff} Hz")

            # Fallback for unhandled or customized metrics
            unhandled = []
            for k, v in metrics.items():
                if k not in handled_keys and not k.startswith("dataset_profiler.") and not k.endswith(".unit"):
                    if isinstance(v, (int, float, str, bool)):
                        clean_k = k.split(".")[-1].replace("_", " ")
                        unhandled.append(f"{clean_k}: {v}")
            if unhandled:
                metric_descriptions.append(", ".join(unhandled))

            if metric_descriptions:
                lines.append(f"Measured values: {'; '.join(metric_descriptions)}.")

        # Disclose any warnings
        if evidence.warnings:
            lines.append(f"Note: {'; '.join(evidence.warnings)}.")

        if not lines:
            lines.append("Acoustic analysis completed with no anomalous features detected.")

        return " ".join(lines)


