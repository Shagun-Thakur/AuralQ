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
5. DATASET AUDITING VS SINGLE-TRACK REMEDIATION:
   - If the query asks about auditing, profiling, or preprocessing a multi-file DATASET or directory (e.g. 'audit this dataset', 'profile dataset', 'dataset preprocessing pipeline', 'give dataset report'), select the `dataset_profiler` tool.
   - If the query asks about cleaning, filtering, fixing, or preprocessing a SINGLE AUDIO RECORDING (e.g. 'what preprocessing steps to reduce noise?', 'how should I clean this audio?', 'what filter to use?'), do NOT select `dataset_profiler`! Instead, select the single-audio diagnostic tools: `spectral_flatness` (noise profile), `snr_estimate` (signal-to-noise ratio), `rms_energy` (loudness/headroom), and/or `spectral_centroid` (frequency balance).
6. APPLIED AUDIO USE-CASE QUESTIONS ARE VALID (DO NOT ABSTAIN):
   - Questions evaluating an audio recording for a practical use-case (e.g. 'Can I use this for a vlog background?', 'Is this suitable for a podcast intro?', 'Will this clash with speech?', 'Is this audio clean enough for voiceover?') ARE acoustic engineering questions and MUST NOT be abstained.
   - Select the tools that measure the physical properties determining suitability:
     * Background audio / speech masking: Select `rms_energy` (for dialogue masking headroom) and `spectral_centroid` (for treble clashing with speech sibilance).
     * Noise reduction / cleaning recommendations: Select `spectral_flatness` and `snr_estimate`.
     * Vocal register / melody: Select `pitch_yin`.
     * Percussive transients / alarms: Select `zero_crossing_rate` and `spectral_centroid`.
   - SET "is_abstention": true ONLY if the question has NO connection to audio physics or sound engineering (e.g. asking for artist identity, lyric meanings, medical diagnoses, stock predictions, cooking recipes, or general non-audio trivia).
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
    dataset_path: Optional[str] = None,
) -> str:
    """Construct the user prompt incorporating audio metadata and dataset context if available."""
    prompt_parts = []
    if audio_metadata:
        prompt_parts.append("AUDIO CONTEXT:")
        for k, v in audio_metadata.items():
            prompt_parts.append(f"- {k}: {v}")
        prompt_parts.append("")

    if dataset_path:
        prompt_parts.append("DATASET CONTEXT:")
        prompt_parts.append(f"- Active Dataset Directory: {dataset_path}")
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

