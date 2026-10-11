"""System prompts and prompt engineering templates for Layer 2 LLM orchestration."""

import json
from typing import Any, Dict, List, Optional
from src.tools.registry import list_tools_catalogue


def format_tool_catalogue_for_prompt() -> str:
    """Format registered tools with their parameters and keywords into markdown for LLM context."""
    catalogue = list_tools_catalogue()
    lines = []
    for tool in catalogue:
        lines.append(f"### Tool: `{tool['name']}` ({tool['category']})")
        lines.append(f"- **Description**: {tool['description']}")
        lines.append(f"- **Intent Keywords**: {', '.join(tool['intent_keywords'])}")
        
        props = tool['parameters'].get('properties', {})
        req = tool['parameters'].get('required', [])
        if props:
            lines.append("- **Parameters**:")
            for p_name, p_info in props.items():
                p_type = p_info.get('type', 'any')
                p_default = p_info.get('default', 'None')
                p_desc = p_info.get('description', '')
                req_str = " (Required)" if p_name in req else f" (Optional, default: {p_default})"
                lines.append(f"  * `{p_name}` ({p_type}){req_str}: {p_desc}")
        else:
            lines.append("- **Parameters**: None required.")
        lines.append("")
    return "\n".join(lines)


SYSTEM_INTENT_PROMPT = """You are AuralQ's Intent Parsing and Tool Selection Engine.
Your sole responsibility is to translate a user's natural language question about an audio recording or dataset into a minimal, sufficient set of deterministic DSP tool calls.

CRITICAL ARCHITECTURAL RULES:
1. You are NEVER an acoustic compute engine. Never guess or fabricate frequency, energy, or dB numbers.
2. Select ONLY tools from the registered tool catalogue below. Never invent tool names.
3. Select the MINIMUM SUFFICIENT set of tools needed to answer the question. Maximum 3 tools per query.
4. If a question asks about multiple properties (e.g. loudness AND brightness), include all required tools (up to 3).
5. If the query asks about a dataset folder or preprocessing recommendations, select the `dataset_profiler` tool.
6. If the question cannot be answered by acoustic signal processing (e.g. asking for singer identity, song lyrics, medical diagnosis, or general trivia unrelated to audio physics), set "is_abstention": true and provide a clear "abstention_reason".
7. You must respond in STRICT VALID JSON format conforming to the schema below. No conversational prose outside the JSON.

OUTPUT JSON SCHEMA:
{
  "user_intent": "string (brief summary of what acoustic property or action the user seeks)",
  "reasoning": "string (why the selected tools or abstention are appropriate)",
  "selected_tools": [
    {
      "tool_name": "exact_tool_name_from_catalogue",
      "parameters": {}
    }
  ],
  "is_abstention": false,
  "abstention_reason": null,
  "requires_clarification": false,
  "clarification_question": null
}

REGISTERED TOOL CATALOGUE:
"""


def build_system_prompt() -> str:
    """Build the complete system prompt with dynamically injected tool catalogue."""
    tool_docs = format_tool_catalogue_for_prompt()
    return f"{SYSTEM_INTENT_PROMPT}\n{tool_docs}"


def build_user_prompt(
    question: str,
    audio_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Construct the user prompt incorporating audio metadata context if available."""
    prompt_parts = []
    if audio_metadata:
        prompt_parts.append("AUDIO CONTEXT:")
        for k, v in audio_metadata.items():
            prompt_parts.append(f"- {k}: {v}")
        prompt_parts.append("")

    prompt_parts.append(f"USER QUESTION: {question.strip()}")
    prompt_parts.append("\nReturn your analysis and tool selection strictly as valid JSON:")
    return "\n".join(prompt_parts)


def build_correction_prompt(
    original_output: str,
    validation_error: str,
) -> str:
    """Construct feedback prompt for structured retry when schema validation fails."""
    return f"""Your previous response was invalid JSON or violated the Pydantic schema constraints.

VALIDATION ERROR:
{validation_error}

ORIGINAL RESPONSE:
{original_output}

Please fix the error and output strictly valid JSON conforming to the IntentParseResult schema.
Do not include any markdown formatting, greetings, or explanations outside the JSON object.
"""

