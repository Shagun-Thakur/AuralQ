"""Local Ollama HTTP client wrapper with VRAM-safe settings and structured retry."""

import json
import re
import time
from typing import Any, Dict, Optional
from pydantic import ValidationError
import requests

from src.llm.prompts import (
    build_correction_prompt,
    build_system_prompt,
    build_user_prompt,
)
from src.llm.schemas import (
    IntentParseResult,
    OllamaGenerationMetadata,
    OllamaResponse,
)
from src.utils.config import get_models_config


def clean_json_text(text: str) -> str:
    """Strip markdown code fence blocks if present in LLM response."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Match outermost braces if text has leading/trailing characters
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    return text


class OllamaClient:
    """
    Client for local Ollama server enforcing hardware-safe VRAM boundaries
    and self-healing Pydantic schema validation retries.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        num_ctx: int = 2048,
        temperature: float = 0.0,
        timeout_sec: float = 30.0,
        max_retries: int = 2,
    ):
        cfg = get_models_config()
        primary_cfg = cfg.get("primary", {})
        
        self.model_name = model_name or primary_cfg.get("name", "llama3.2:3b-instruct-q4_K_M")
        self.endpoint = endpoint or primary_cfg.get("endpoint", "http://localhost:11434/api/generate")
        self.num_ctx = num_ctx or primary_cfg.get("num_ctx", 2048)
        self.temperature = temperature if temperature is not None else primary_cfg.get("temperature", 0.0)
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def _call_ollama_raw(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        """Execute raw HTTP POST request to Ollama with hardlocked GPU context options."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
            },
        }
        if system:
            payload["system"] = system

        response = requests.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        return response.json()

    def parse_intent(
        self,
        question: str,
        audio_metadata: Optional[Dict[str, Any]] = None,
    ) -> OllamaResponse:
        """
        Parse user question into single-shot tool intent with self-healing validation retry.
        Guarantees structured output or explicit error status.
        """
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(question, audio_metadata=audio_metadata)
        current_prompt = user_prompt
        current_system = system_prompt

        total_time_ms = 0.0
        prompt_tokens = 0
        eval_tokens = 0
        raw_text = ""
        last_error = ""

        for attempt in range(self.max_retries + 1):
            t_start = time.perf_counter()
            try:
                res_json = self._call_ollama_raw(current_prompt, system=current_system)
            except requests.exceptions.RequestException as req_err:
                return OllamaResponse(
                    is_valid=False,
                    status="connection_error",
                    error_message=f"Failed to connect to Ollama at {self.endpoint}: {str(req_err)}",
                    retry_count=attempt,
                )
            except Exception as e:
                return OllamaResponse(
                    is_valid=False,
                    status="connection_error",
                    error_message=f"Unexpected error communicating with Ollama: {str(e)}",
                    retry_count=attempt,
                )

            t_elapsed = (time.perf_counter() - t_start) * 1000
            total_time_ms += t_elapsed

            # Accumulate telemetry
            prompt_tokens += res_json.get("prompt_eval_count", 0)
            eval_tokens += res_json.get("eval_count", 0)
            raw_text = res_json.get("response", "")

            # Attempt Pydantic validation
            try:
                cleaned = clean_json_text(raw_text)
                parsed = IntentParseResult.model_validate_json(cleaned)

                telemetry = OllamaGenerationMetadata(
                    model=self.model_name,
                    total_duration_ms=round(total_time_ms, 2),
                    load_duration_ms=round(res_json.get("load_duration", 0) / 1e6, 2),
                    prompt_eval_count=prompt_tokens,
                    eval_count=eval_tokens,
                    eval_duration_ms=round(res_json.get("eval_duration", 0) / 1e6, 2),
                )

                status = "abstained" if parsed.is_abstention else "success"
                return OllamaResponse(
                    is_valid=True,
                    status=status,
                    raw_text=raw_text,
                    parsed_intent=parsed,
                    retry_count=attempt,
                    metadata=telemetry,
                )

            except (ValidationError, json.JSONDecodeError) as val_err:
                last_error = str(val_err)
                if attempt < self.max_retries:
                    # Construct structured correction prompt for retry
                    current_prompt = build_correction_prompt(raw_text, last_error)
                    current_system = None  # Ollama retains system instructions better in conversation

        telemetry = OllamaGenerationMetadata(
            model=self.model_name,
            total_duration_ms=round(total_time_ms, 2),
            prompt_eval_count=prompt_tokens,
            eval_count=eval_tokens,
        )
        return OllamaResponse(
            is_valid=False,
            status="schema_error",
            raw_text=raw_text,
            retry_count=self.max_retries,
            error_message=f"Pydantic schema validation failed after {self.max_retries} retries: {last_error}",
            metadata=telemetry,
        )

