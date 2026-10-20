"""Pydantic v2 schemas for Layer 4: Evidence Packaging, Turn History, and Agent State."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from src.audio.validation import AudioMetadata
from src.tools.schemas import ToolResultModel
from src.llm.schemas import IntentParseResult


class EvidencePackage(BaseModel):
    """
    Consolidated, tamper-evident acoustic evidence package.
    Aggregates deterministic DSP measurements, perceptual anchors, and citation aliases.
    """
    query: str = Field(description="The original natural language question from the user.")
    audio_id: str = Field(default="audio_input", description="Identifier or filename of the audio under analysis.")
    audio_metadata: Optional[AudioMetadata] = Field(default=None, description="Layer 1 audio hygiene and format metadata.")
    status: Literal["SUCCESS", "PARTIAL", "TOOL_FAILURE", "ABSTAIN"] = "SUCCESS"
    tools_called: List[str] = Field(default_factory=list, description="Names of tools dispatched.")
    tool_results: List[ToolResultModel] = Field(default_factory=list, description="Raw structured results from DSP tools.")
    consolidated_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Unified dictionary mapping metric names to numerical values.",
    )
    consolidated_anchors: Dict[str, str] = Field(
        default_factory=dict,
        description="Deterministic qualitative perceptual anchors.",
    )
    all_citation_aliases: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Lookup map of canonical numeric tokens for downstream regex validation.",
    )
    total_dsp_latency_ms: float = 0.0
    warnings: List[str] = Field(default_factory=list, description="Warnings aggregated from ingestion and DSP layers.")
    abstention_reason: Optional[str] = Field(default=None, description="Reason if system abstains from answering.")

    def format_evidence_for_llm(self) -> str:
        """Format the evidence package as structured markdown for Layer 2/5 generation."""
        if self.status == "ABSTAIN":
            return f"<evidence>\nSTATUS: ABSTAIN\nREASON: {self.abstention_reason or 'Insufficient evidence'}\n</evidence>"

        lines = ["<evidence>"]
        lines.append(f"STATUS: {self.status}")
        if self.audio_metadata:
            lines.append(f"AUDIO_FILE: {self.audio_metadata.file_path}")
            lines.append(f"DURATION: {self.audio_metadata.duration_sec}s | SAMPLE_RATE: {self.audio_metadata.sample_rate}Hz")
            if self.audio_metadata.warnings:
                lines.append(f"INGESTION_WARNINGS: {'; '.join(self.audio_metadata.warnings)}")

        lines.append("\nMEASURED_METRICS:")
        for k, v in self.consolidated_metrics.items():
            lines.append(f"- {k}: {v}")

        if self.consolidated_anchors:
            lines.append("\nPERCEPTUAL_ANCHORS:")
            for k, v in self.consolidated_anchors.items():
                lines.append(f"- {k}: {v}")

        if self.warnings:
            lines.append(f"\nDSP_WARNINGS: {'; '.join(self.warnings)}")

        lines.append("</evidence>")
        return "\n".join(lines)


class CitationMatch(BaseModel):
    """Details of an individual numerical token identified and checked in the response."""
    raw_token: str = Field(description="The exact numeric substring extracted from the response.")
    parsed_value: Optional[float] = Field(default=None, description="Numeric float representation of token.")
    matched_metric: Optional[str] = Field(default=None, description="Canonical metric or alias this matched against.")
    match_type: Literal["exact_alias", "tolerance_metric", "metadata", "query_echo", "engineering_reference"] = "exact_alias"
    is_grounded: bool = True


class GroundingReport(BaseModel):
    """Deterministic audit report for numeric citation grounding in an agent response."""
    is_fully_grounded: bool = Field(description="True if all numeric claims are grounded in evidence.")
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Proportion of verified numbers (0.0 to 1.0).")
    total_numbers_cited: int = 0
    verified_citations: List[CitationMatch] = Field(default_factory=list)
    hallucinated_numbers: List[str] = Field(default_factory=list, description="Unverified numerical tokens.")
    was_repaired: bool = Field(default=False, description="True if response was substituted with deterministic fallback.")
    original_response_text: Optional[str] = Field(default=None, description="Original ungrounded text if repaired.")


class TurnRecord(BaseModel):
    """Encapsulates a single completed conversational turn in an agent session."""
    turn_id: int = Field(description="Sequential 1-based index of this turn.")
    timestamp: float = Field(description="Unix timestamp when this turn was completed.")
    query: str = Field(description="User natural-language query.")
    parsed_intent: Optional[IntentParseResult] = Field(default=None, description="Layer 2 intent parsing output.")
    evidence: EvidencePackage = Field(description="Structured evidence package produced for this turn.")
    response_text: str = Field(description="Final synthesized, grounded response delivered to user.")
    latency_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing breakdown in milliseconds (intent_ms, dsp_ms, synthesis_ms, total_ms).",
    )
    cached_tools_used: List[str] = Field(
        default_factory=list,
        description="Names of tools whose results were retrieved from session cache without DSP re-computation.",
    )
    grounding_report: Optional[GroundingReport] = Field(
        default=None,
        description="Deterministic citation verification and hallucination audit.",
    )


class AgentResponse(BaseModel):
    """User-facing output of a single agent query turn."""
    query: str = Field(description="User query.")
    response_text: str = Field(description="Grounded natural-language answer.")
    status: Literal["success", "abstained", "dsp_error", "schema_error", "connection_error"] = "success"
    evidence: EvidencePackage = Field(description="The underlying evidence package.")
    turn_record: TurnRecord = Field(description="Complete record of this turn for history.")
    latency_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Latency metrics for this turn (ms).",
    )
    cached_tools_used: List[str] = Field(
        default_factory=list,
        description="Tools resolved from cache.",
    )
    grounding_report: Optional[GroundingReport] = Field(
        default=None,
        description="Deterministic numeric citation grounding report.",
    )


class SessionSummary(BaseModel):
    """Metadata summary of an ongoing AuralQ agent session."""
    session_id: str
    turn_count: int
    active_audio_path: Optional[str] = None
    active_dataset_path: Optional[str] = None
    cached_tools: List[str] = Field(default_factory=list)
