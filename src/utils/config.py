"""Configuration loader utility for AuralQ."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_yaml(file_name: str) -> Dict[str, Any]:
    """Load a YAML config file from the configs directory."""
    path = CONFIGS_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_audio_config() -> Dict[str, Any]:
    """Load audio ingestion configuration."""
    return load_yaml("audio.yaml")


def get_tools_config() -> Dict[str, Any]:
    """Load tools catalogue configuration."""
    return load_yaml("tools.yaml")


def get_models_config() -> Dict[str, Any]:
    """Load model parameters and endpoint configuration."""
    return load_yaml("models.yaml")

