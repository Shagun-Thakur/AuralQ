"""Cloud LLM Client Adapter for AuralQ Public Demo Mode.

Provides drop-in compatibility with OllamaClient for cloud-hosted Streamlit demos
using fast, free-tier cloud APIs (Google Gemini 2.5 Flash / Groq) without requiring
local GPU hardware.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from pydantic import ValidationError

from src.llm.client import OllamaResponse, clean_json_text
from src.llm.prompts import build_system_prompt, build_user_prompt
from src.llm.schemas import IntentParseResult


class CloudClient:
    """
    Cloud LLM client providing unified intent parsing and text synthesis
    for the AuralQ public web demo without local GPU dependencies.
    """

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_sec: float = 30.0,
    ):
        self.provider = provider.lower()
        # Check Streamlit secrets if available, then environment variables
        st_key = None
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                if self.provider == "gemini":
                    st_key = st.secrets.get("GEMINI_API_KEY")
                elif self.provider == "groq":
                    st_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass

        self.api_key = (
            api_key
            or st_key
            or (os.environ.get("GEMINI_API_KEY") if self.provider == "gemini" else os.environ.get("GROQ_API_KEY"))
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or ""
        )
        self.timeout_sec = timeout_sec


        if self.provider == "gemini":
            self.model_name = model_name or "gemini-2.5-flash"
        elif self.provider == "groq":
            self.model_name = model_name or "llama-3.1-8b-instant"
        else:
            self.model_name = model_name or "default"

    def _call_gemini_rest(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Google Gemini REST API directly via standard library."""
        if not self.api_key:
            raise ValueError("No Gemini API key provided. Please supply an API key in the Cloud Demo settings.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        body: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json" if "JSON" in (system_prompt or "") else "text/plain",
            },
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        req_data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        raise RuntimeError("Empty response received from Gemini API.")

    def _call_groq_rest(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Groq OpenAI-compatible REST API directly via standard library."""
        if not self.api_key:
            raise ValueError("No Groq API key provided. Please supply an API key in the Cloud Demo settings.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
        }
        req_data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "")
        raise RuntimeError("Empty response received from Groq API.")

    def generate_raw(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Route to appropriate cloud API provider."""
        if self.provider == "gemini":
            return self._call_gemini_rest(prompt, system_prompt)
        elif self.provider == "groq":
            return self._call_groq_rest(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported cloud provider: {self.provider}")

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Synthesize natural-language text matching OllamaClient interface."""
        t_start = time.perf_counter()
        try:
            raw_text = self.generate_raw(prompt, system_prompt)
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {"response": raw_text, "duration_ms": duration_ms}
        except Exception as e:
            return {"response": f"Cloud API error ({self.provider}): {str(e)}", "duration_ms": 0.0}

    def parse_intent(
        self,
        query: str,
        audio_metadata: Optional[Dict[str, Any]] = None,
        dataset_path: Optional[str] = None,
        max_retries: int = 1,
    ) -> OllamaResponse:
        """Parse acoustic intent into IntentParseResult matching OllamaClient interface."""
        t_start = time.perf_counter()
        user_prompt = build_user_prompt(query, audio_metadata, dataset_path=dataset_path)
        system_prompt = build_system_prompt()

        for attempt in range(max_retries + 1):
            try:
                raw_reply = self.generate_raw(user_prompt, system_prompt)
                clean_json = clean_json_text(raw_reply)
                data = json.loads(clean_json)
                parsed = IntentParseResult.model_validate(data)
                duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
                return OllamaResponse(
                    raw_response=raw_reply,
                    parsed_intent=parsed,
                    is_valid=True,
                    status="success",
                    duration_ms=duration_ms,
                    retry_count=attempt,
                )
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt < max_retries:
                    user_prompt += f"\n\nSCHEMA ERROR: Your previous response failed schema validation: {str(e)}. Output ONLY valid JSON matching the schema."
                    continue
                duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
                return OllamaResponse(
                    raw_response="",
                    is_valid=False,
                    status="schema_error",
                    error_message=f"Cloud schema validation error: {str(e)}",
                    duration_ms=duration_ms,
                    retry_count=attempt,
                )
            except Exception as e:
                duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
                return OllamaResponse(
                    raw_response="",
                    is_valid=False,
                    status="connection_error",
                    error_message=f"Cloud connection error: {str(e)}",
                    duration_ms=duration_ms,
                    retry_count=attempt,
                )

