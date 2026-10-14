"""Ollama Model Discovery and Service Health Helper for AuralQ Cockpit."""

import json
import urllib.error
import urllib.request
from typing import List, Optional


def check_ollama_status(endpoint: str = "http://127.0.0.1:11434") -> bool:
    """Check if local Ollama daemon is reachable."""
    try:
        req = urllib.request.Request(f"{endpoint}/api/tags", headers={"User-Agent": "AuralQ"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_installed_ollama_models(endpoint: str = "http://127.0.0.1:11434") -> List[str]:
    """
    Query Ollama's local tags API to discover all installed models on the user's system.
    Returns a sorted list of model names, prioritizing llama3.2 variants.
    """
    try:
        req = urllib.request.Request(f"{endpoint}/api/tags", headers={"User-Agent": "AuralQ"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                # Sort: put llama3.2 at the top if present
                models.sort(key=lambda m: (0 if "llama3.2" in m.lower() else 1, m))
                return models
    except Exception:
        pass
    return []


def pull_ollama_model(model_name: str, endpoint: str = "http://127.0.0.1:11434") -> bool:
    """
    Trigger model pull via Ollama REST API.
    """
    try:
        req_data = json.dumps({"name": model_name, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{endpoint}/api/pull",
            data=req_data,
            headers={"Content-Type": "application/json", "User-Agent": "AuralQ"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            return resp.status == 200
    except Exception:
        return False

