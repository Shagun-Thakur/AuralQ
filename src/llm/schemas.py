"""Pydantic v2 schemas for Layer 2: LLM Interface, Intent Parsing, and Tool Dispatch."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from src.tools.registry import get_registered_tool_names


class ToolCallIntent(BaseModel):
    """Specification of an individual tool call selected by the LLM."""
    tool_name: str = Field(description="Exact registered name of the tool to execute.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Arguments dictionary matching the tool's parameter schema.")

    @field_validator("tool_name")
    def validate_tool_name(cls, v: str) -> str:
        allowed = get_registered_tool_names()
        if v not in allowed:
            raise ValueError(f"Tool '{v}' is not in the registered tool catalogue. Allowed: {allowed}")
        return v


class IntentParseResult(BaseModel):
    """
    Structured outcome of the LLM single-shot intent parsing phase.
    Conforms to strict single-shot execution: at most 3 tools selected.
    """
    user_intent: str = Field(description="Concise description of the acoustic property or action the user is querying.")
    reasoning: str = Field(description="Short rationale for selecting the specified tools.")
    selected_tools: List[ToolCallIntent] = Field(
        default_factory=list,
        max_length=3,
        description="List of tools to execute (minimum sufficient set, maximum 3 tools).",
    )
    is_abstention: bool = Field(
        default=False,
        description="Set to true if the question cannot be answered by acoustic analysis or is out of scope.",
    )
    abstention_reason: Optional[str] = Field(
        default=None,
        description="Explanation if the system must abstain from answering.",
    )
    requires_clarification: bool = Field(
        default=False,
        description="Set to true if the user query is excessively ambiguous.",
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Clarification prompt for the user if requires_clarification is true.",
    )


class OllamaGenerationMetadata(BaseModel):
    """Execution telemetry captured from local Ollama inference."""
    model: str
    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0
    prompt_eval_count: int = 0
    prompt_eval_duration_ms: float = 0.0
    eval_count: int = 0
    eval_duration_ms: float = 0.0


class OllamaResponse(BaseModel):
    """Universal envelope for Layer 2 LLM inference results."""
    is_valid: bool
    status: Literal["success", "schema_error", "connection_error", "abstained"] = "success"
    raw_text: str = ""
    parsed_intent: Optional[IntentParseResult] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    metadata: Optional[OllamaGenerationMetadata] = None

