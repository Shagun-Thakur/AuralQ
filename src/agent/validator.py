"""Layer 4 / 5: Deterministic Numeric Citation Validator and Hallucination Auditor.

Performs rule-based token extraction, tolerance comparison, and evidence grounding checks.
Zero secondary LLM calls; 100% deterministic regex and numerical validation.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.agent.schemas import CitationMatch, EvidencePackage, GroundingReport
from src.agent.synthesizer import ResponseSynthesizer


# Regex capturing numeric tokens with optional units (Hz, kHz, dB, dBFS, %, s, ms, files, channels)
# Handles signed numbers, decimals, scientific notation, and percentages.
NUMERIC_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?%?(?:\s*(?:Hz|kHz|dBFS|dB|seconds|second|sec|s|ms|files|file|channels|channel|hours|hour))?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

UNIT_SPLIT_PATTERN = re.compile(
    r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?%?)\s*([A-Za-z%]*)$",
    re.IGNORECASE,
)


def parse_numeric_candidate(token: str) -> Tuple[Optional[float], str]:
    """
    Split a candidate token like '1520.45 Hz' or '-12.4dBFS' or '85%'
    into a numerical float value and a normalized unit string.
    """
    token_clean = token.strip().rstrip(".,;:!?)]}\"'")
    m = UNIT_SPLIT_PATTERN.match(token_clean)
    if not m:
        return None, ""

    num_part, unit_part = m.group(1), m.group(2)
    is_percent = False
    if num_part.endswith("%"):
        is_percent = True
        num_part = num_part[:-1]
        unit_part = "%"

    try:
        val = float(num_part)
        return val, unit_part.lower()
    except (ValueError, TypeError):
        return None, ""


class CitationValidator:
    """
    Deterministic citation validator verifying all numbers in an agent's response
    against the underlying Layer 3 EvidencePackage.
    """

    def __init__(
        self,
        rel_tolerance: float = 0.05,  # 5% relative error margin
        abs_tolerance: float = 0.5,   # For whole-number roundoffs (e.g. 1520 vs 1520.4)
    ):
        self.rel_tolerance = rel_tolerance
        self.abs_tolerance = abs_tolerance

    def extract_numeric_tokens(self, text: str) -> List[str]:
        """Extract all candidate numerical tokens from text using regex."""
        if not text:
            return []
        # Strip markdown list markers (e.g. "\n1. ", "\n2. ", "^1. ") so list numbering is not treated as numeric claims
        cleaned_text = re.sub(r"(?:^|\n)\s*\d+\.\s+", "\n", text)
        matches = NUMERIC_TOKEN_PATTERN.findall(cleaned_text)
        # Strip trailing punctuation attached to matches
        clean_tokens = [m.strip().rstrip(".,;:!?)]}\"'") for m in matches if m.strip()]
        return clean_tokens


    def validate(
        self,
        response_text: str,
        evidence: EvidencePackage,
        query: str = "",
    ) -> GroundingReport:
        """
        Verify whether all numerical claims in response_text are grounded in evidence.
        Returns a structured GroundingReport.
        """
        # If the query was an explicit abstention, no DSP measurements are required
        if evidence.status == "ABSTAIN":
            return GroundingReport(
                is_fully_grounded=True,
                grounding_score=1.0,
                total_numbers_cited=0,
                verified_citations=[],
                hallucinated_numbers=[],
            )

        extracted_tokens = self.extract_numeric_tokens(response_text)
        if not extracted_tokens:
            # Qualitative-only response without numbers is 100% grounded
            return GroundingReport(
                is_fully_grounded=True,
                grounding_score=1.0,
                total_numbers_cited=0,
                verified_citations=[],
                hallucinated_numbers=[],
            )

        query_tokens = set(self.extract_numeric_tokens(query))

        # Flatten all registered aliases into a lookup set (normalized lowercase)
        alias_to_metric: Dict[str, str] = {}
        for metric_name, aliases in evidence.all_citation_aliases.items():
            for a in aliases:
                alias_to_metric[a.strip().lower()] = metric_name

        # Extract numeric metrics from evidence (supporting scalars, nested dictionaries, and recommendation lists)
        numeric_metrics: Dict[str, float] = {}

        def _extract_nested_numerics(prefix: str, val: Any) -> None:
            if isinstance(val, (int, float)) and not isinstance(val, bool) and not math.isnan(val) and not math.isinf(val):
                numeric_metrics[prefix] = float(val)
            elif isinstance(val, str):
                for m_tok in NUMERIC_TOKEN_PATTERN.findall(val):
                    v_num, _ = parse_numeric_candidate(m_tok)
                    if v_num is not None:
                        numeric_metrics[f"{prefix}.embedded_{v_num}"] = v_num
            elif isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    # If key itself is numeric (e.g. sample rates 22050, 16000 or channel counts 1, 2)
                    try:
                        numeric_metrics[f"{prefix}.key_{sub_k}"] = float(sub_k)
                    except (ValueError, TypeError):
                        pass
                    _extract_nested_numerics(f"{prefix}.{sub_k}", sub_v)
            elif isinstance(val, (list, tuple)):
                for idx, item in enumerate(val):
                    if isinstance(item, dict):
                        for item_k, item_v in item.items():
                            if item_k == "canonical_citations" and isinstance(item_v, list):
                                for cit in item_v:
                                    if isinstance(cit, str) and cit.strip():
                                        alias_to_metric[cit.strip().lower()] = f"{prefix}[{idx}].canonical_citation"
                            else:
                                _extract_nested_numerics(f"{prefix}[{idx}].{item_k}", item_v)
                    elif isinstance(item, (int, float, str, dict, list)):
                        _extract_nested_numerics(f"{prefix}[{idx}]", item)



        for k, v in evidence.consolidated_metrics.items():
            _extract_nested_numerics(k, v)


        # Extract audio metadata numbers
        metadata_numbers: Dict[str, float] = {}
        if evidence.audio_metadata:
            meta = evidence.audio_metadata
            metadata_numbers["metadata.sample_rate"] = float(meta.sample_rate)
            metadata_numbers["metadata.duration_sec"] = float(meta.duration_sec)
            metadata_numbers["metadata.num_channels"] = float(meta.num_channels)
            metadata_numbers["metadata.num_samples"] = float(meta.num_samples)
            metadata_numbers["metadata.rms_dbfs"] = float(meta.rms_dbfs)
            metadata_numbers["metadata.clipping_ratio"] = float(meta.clipping_ratio)
            metadata_numbers["metadata.clipping_percentage"] = float(meta.clipping_ratio * 100.0)

        verified_citations: List[CitationMatch] = []
        hallucinated_numbers: List[str] = []

        for token in extracted_tokens:
            token_clean = token.strip()
            token_lower = token_clean.lower()
            val, unit = parse_numeric_candidate(token_clean)

            # Check 1: Query echo (number appeared in the user's prompt)
            if token_clean in query_tokens:
                verified_citations.append(
                    CitationMatch(
                        raw_token=token_clean,
                        parsed_value=val,
                        matched_metric="query_prompt",
                        match_type="query_echo",
                        is_grounded=True,
                    )
                )
                continue

            # Check 2: Exact token alias match in evidence
            if token_lower in alias_to_metric:
                metric_name = alias_to_metric[token_lower]
                verified_citations.append(
                    CitationMatch(
                        raw_token=token_clean,
                        parsed_value=val,
                        matched_metric=metric_name,
                        match_type="exact_alias",
                        is_grounded=True,
                    )
                )
                continue

            # Check 3: Check bare numeric string against aliases (without unit)
            if val is not None:
                bare_num_str = str(int(round(val))) if val.is_integer() else f"{val}"
                if bare_num_str in alias_to_metric:
                    metric_name = alias_to_metric[bare_num_str]
                    verified_citations.append(
                        CitationMatch(
                            raw_token=token_clean,
                            parsed_value=val,
                            matched_metric=metric_name,
                            match_type="exact_alias",
                            is_grounded=True,
                        )
                    )
                    continue

            # Check 4: Numerical tolerance comparison against consolidated metrics
            matched = False
            if val is not None:
                all_targets = {**numeric_metrics, **metadata_numbers}
                for target_name, target_val in all_targets.items():
                    if self._is_numeric_match(val, target_val, unit):
                        m_type = "metadata" if target_name.startswith("metadata.") else "tolerance_metric"
                        verified_citations.append(
                            CitationMatch(
                                raw_token=token_clean,
                                parsed_value=val,
                                matched_metric=target_name,
                                match_type=m_type,
                                is_grounded=True,
                            )
                        )
                        matched = True
                        break

            if not matched:
                hallucinated_numbers.append(token_clean)

        total_cited = len(extracted_tokens)
        verified_count = len(verified_citations)
        score = round(verified_count / max(total_cited, 1), 4)
        is_fully_grounded = (len(hallucinated_numbers) == 0)

        return GroundingReport(
            is_fully_grounded=is_fully_grounded,
            grounding_score=score,
            total_numbers_cited=total_cited,
            verified_citations=verified_citations,
            hallucinated_numbers=hallucinated_numbers,
            was_repaired=False,
        )

    def _is_numeric_match(self, candidate: float, target: float, unit: str = "") -> bool:
        """Evaluate numeric candidate against ground truth target under tolerances and conversions."""
        # Scale absolute tolerance according to magnitude of target
        effective_abs_tol = self.abs_tolerance if abs(target) >= 10.0 else (0.01 if abs(target) >= 1.0 else 0.005)

        # Direct equivalence
        if math.isclose(candidate, target, rel_tol=self.rel_tolerance, abs_tol=effective_abs_tol):
            return True

        # kHz to Hz conversion (e.g. 1.5 kHz cited for 1500 Hz target)
        if unit in ["khz"]:
            if math.isclose(candidate * 1000.0, target, rel_tol=self.rel_tolerance, abs_tol=self.abs_tolerance):
                return True
        elif not unit and candidate >= 1.0 and target >= 1000.0:
            if math.isclose(candidate * 1000.0, target, rel_tol=self.rel_tolerance, abs_tol=self.abs_tolerance):
                return True

        # Percentage conversion: ONLY when unit is explicitly '%'
        if unit == "%":
            # Compare candidate/100 to target (e.g. 85% to 0.85)
            if math.isclose(candidate / 100.0, target, rel_tol=self.rel_tolerance, abs_tol=0.01):
                return True
            # Or if target was already stored as a percentage (e.g. 85.0)
            if math.isclose(candidate, target, rel_tol=self.rel_tolerance, abs_tol=effective_abs_tol):
                return True

        # Hours to seconds conversion
        if unit in ["hours", "hour"]:
            if math.isclose(candidate * 3600.0, target, rel_tol=self.rel_tolerance, abs_tol=effective_abs_tol):
                return True
        elif unit in ["s", "sec", "seconds", "second"]:
            if math.isclose(candidate / 3600.0, target, rel_tol=self.rel_tolerance, abs_tol=0.001):
                return True

        return False

    def repair_if_unverified(
        self,
        response_text: str,
        evidence: EvidencePackage,
        query: str = "",
        synthesizer: Optional[ResponseSynthesizer] = None,
    ) -> Tuple[str, GroundingReport]:
        """
        Audit response_text. If any ungrounded numeric citations are detected,
        substitute with the deterministic evidence fallback response.
        """
        report = self.validate(response_text, evidence, query)

        if report.is_fully_grounded:
            return response_text, report

        # Hallucination detected: perform deterministic fallback substitution
        synth = synthesizer or ResponseSynthesizer()
        fallback_text = synth._deterministic_fallback_synthesis(evidence)
        repaired_text = (
            f"{fallback_text} "
            "[Verified DSP Grounding: Response was deterministically repaired to eliminate unverified numerical claims.]"
        )
        report.was_repaired = True
        report.original_response_text = response_text
        return repaired_text, report
