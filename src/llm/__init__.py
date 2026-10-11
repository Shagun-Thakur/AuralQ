"""Layer 2: LLM orchestration, intent parsing, and tool-calling protocol for AuralQ."""

from src.llm.client import OllamaClient, clean_json_text
from src.llm.prompts import (
    build_correction_prompt,
    build_system_prompt,
    build_user_prompt,
    format_tool_catalogue_for_prompt,
)
from src.llm.schemas import (
    IntentParseResult,
    OllamaGenerationMetadata,
    OllamaResponse,
    ToolCallIntent,
)

__all__ = [
    "OllamaClient",
    "clean_json_text",
    "build_system_prompt",
    "build_user_prompt",
    "build_correction_prompt",
    "format_tool_catalogue_for_prompt",
    "IntentParseResult",
    "ToolCallIntent",
    "OllamaGenerationMetadata",
    "OllamaResponse",
]

