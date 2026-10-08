"""Layer 1: Audio ingestion, hygiene, and validation."""

from src.audio.loader import AudioLoadResult, load_audio_file
from src.audio.validation import (
    AudioMetadata,
    ValidationResult,
    apply_dc_block,
    check_clipping,
    check_silence,
    compute_rms_dbfs,
    validate_duration,
    validate_format,
)

__all__ = [
    "AudioLoadResult",
    "load_audio_file",
    "AudioMetadata",
    "ValidationResult",
    "apply_dc_block",
    "check_clipping",
    "check_silence",
    "compute_rms_dbfs",
    "validate_duration",
    "validate_format",
]

