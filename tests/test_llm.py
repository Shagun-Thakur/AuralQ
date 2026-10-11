"""Unit and integration tests for Layer 2: LLM Interface, Intent Parsing, and Ollama Dispatch."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.llm.client import OllamaClient, clean_json_text
from src.llm.prompts import build_system_prompt, build_user_prompt, format_tool_catalogue_for_prompt
from src.llm.schemas import IntentParseResult, ToolCallIntent
from src.tools.registry import get_registered_tool_names


# ---------------------------------------------------------------------------
# 1. Unit Tests (Deterministic / Offline)
# ---------------------------------------------------------------------------

def test_clean_json_text():
    """Verify stripping of markdown fences and stray prose around JSON."""
    raw_markdown = "```json\n{\"user_intent\": \"test\", \"reasoning\": \"r\", \"selected_tools\": []}\n```"
    assert clean_json_text(raw_markdown) == '{"user_intent": "test", "reasoning": "r", "selected_tools": []}'

    with_prose = "Here is your JSON:\n{\"user_intent\": \"test\", \"reasoning\": \"r\", \"selected_tools\": []}\nHope this helps!"
    assert clean_json_text(with_prose) == '{"user_intent": "test", "reasoning": "r", "selected_tools": []}'

    plain = '{"user_intent": "test"}'
    assert clean_json_text(plain) == '{"user_intent": "test"}'


def test_intent_schema_valid():
    """Verify valid IntentParseResult instantiates cleanly."""
    valid_data = {
        "user_intent": "brightness_analysis",
        "reasoning": "Spectral centroid measures perceived brightness.",
        "selected_tools": [
            {"tool_name": "spectral_centroid", "parameters": {"n_fft": 2048}}
        ],
        "is_abstention": False,
    }
    result = IntentParseResult.model_validate(valid_data)
    assert result.user_intent == "brightness_analysis"
    assert len(result.selected_tools) == 1
    assert result.selected_tools[0].tool_name == "spectral_centroid"


def test_intent_schema_invalid_tool_name():
    """Verify unlisted tool names fail Pydantic validation."""
    invalid_data = {
        "user_intent": "fake_test",
        "reasoning": "none",
        "selected_tools": [
            {"tool_name": "non_existent_tool_123", "parameters": {}}
        ],
    }
    with pytest.raises(Exception):
        IntentParseResult.model_validate(invalid_data)


def test_intent_schema_max_tool_budget():
    """Verify that selecting more than 3 tools triggers validation error."""
    exceeded_data = {
        "user_intent": "too_many_tools",
        "reasoning": "testing max 3 tools limit",
        "selected_tools": [
            {"tool_name": "rms_energy", "parameters": {}},
            {"tool_name": "zero_crossing_rate", "parameters": {}},
            {"tool_name": "spectral_centroid", "parameters": {}},
            {"tool_name": "spectral_flatness", "parameters": {}},
        ],
    }
    with pytest.raises(Exception):
        IntentParseResult.model_validate(exceeded_data)


def test_prompt_construction_contains_all_tools():
    """Verify that build_system_prompt dynamically injects all registered tools."""
    prompt = build_system_prompt()
    registered_tools = get_registered_tool_names()
    assert len(registered_tools) >= 9
    for tool_name in registered_tools:
        assert f"`{tool_name}`" in prompt
    assert "MINIMUM SUFFICIENT" in prompt
    assert "Maximum 3 tools" in prompt


def test_client_self_healing_retry():
    """Verify that the client retries on schema error and recovers when corrected."""
    client = OllamaClient()

    invalid_response = {"response": "Not valid JSON at all", "prompt_eval_count": 10, "eval_count": 10}
    valid_response = {
        "response": json.dumps({
            "user_intent": "loudness",
            "reasoning": "RMS energy measures loudness.",
            "selected_tools": [{"tool_name": "rms_energy", "parameters": {}}],
            "is_abstention": False,
        }),
        "prompt_eval_count": 20,
        "eval_count": 30,
        "total_duration": 500000000,
    }

    with patch.object(client, "_call_ollama_raw", side_effect=[invalid_response, valid_response]):
        res = client.parse_intent("Is this recording consistently loud?")
        assert res.is_valid is True
        assert res.retry_count == 1
        assert res.parsed_intent is not None
        assert res.parsed_intent.selected_tools[0].tool_name == "rms_energy"


def test_client_connection_error_graceful_handling():
    """Verify that an unreachable endpoint returns a structured error without crashing."""
    client = OllamaClient(endpoint="http://localhost:9999/api/generate", timeout_sec=0.5)
    res = client.parse_intent("Test question")
    assert res.is_valid is False
    assert res.status == "connection_error"
    assert "Failed to connect" in res.error_message


# ---------------------------------------------------------------------------
# 2. Live Integration Tests (against local Ollama llama3.2:3b-instruct-q4_K_M)
# ---------------------------------------------------------------------------

def is_ollama_reachable() -> bool:
    """Helper to check if local Ollama server is alive."""
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not is_ollama_reachable(), reason="Local Ollama server is not running")
def test_live_ollama_single_tool_dispatch():
    """Verify end-to-end tool selection and latency against local llama3.2:3b."""
    client = OllamaClient()
    res = client.parse_intent("Does this recording sound bright or dull?")

    assert res.is_valid is True
    assert res.status == "success"
    assert res.parsed_intent is not None
    assert len(res.parsed_intent.selected_tools) >= 1
    selected_names = [t.tool_name for t in res.parsed_intent.selected_tools]
    # Check that warm generation evaluation time on RTX 3050 is within budget (< 6000 ms)
    assert res.metadata.eval_duration_ms < 6000.0


@pytest.mark.skipif(not is_ollama_reachable(), reason="Local Ollama server is not running")
def test_live_ollama_abstention():
    """Verify that out-of-scope non-acoustic questions cleanly trigger abstention."""
    client = OllamaClient()
    res = client.parse_intent("Who was the 16th president of the United States?")

    assert res.is_valid is True
    assert res.status == "abstained"
    assert res.parsed_intent.is_abstention is True
    assert res.parsed_intent.abstention_reason is not None
    assert len(res.parsed_intent.selected_tools) == 0


@pytest.mark.skipif(not is_ollama_reachable(), reason="Local Ollama server is not running")
def test_live_ollama_dataset_intent():
    """Verify that dataset profiling queries dispatch the dataset_profiler tool."""
    client = OllamaClient()
    res = client.parse_intent("Please inspect my dataset folder and recommend preprocessing steps.")

    assert res.is_valid is True
    assert res.parsed_intent is not None
    selected_names = [t.tool_name for t in res.parsed_intent.selected_tools]
    assert "dataset_profiler" in selected_names

