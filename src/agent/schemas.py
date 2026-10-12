"""Pydantic v2 schemas for Layer 4: Evidence Packaging and Tool Execution State."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from src.audio.validation import AudioMetadata
from src.tools.schemas import ToolResultModel


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

