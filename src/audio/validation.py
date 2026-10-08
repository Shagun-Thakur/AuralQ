"""Audio hygiene, signal validation, and format verification for AuralQ Layer 1."""

from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field


class AudioMetadata(BaseModel):
    """Metadata container for ingested audio files."""
    file_path: str
    duration_sec: float
    sample_rate: int
    num_channels: int
    num_samples: int
    rms_dbfs: float
    is_clipped: bool = False
    clipping_ratio: float = 0.0
    warnings: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Structured response for audio validation checks."""
    is_valid: bool
    status: Literal["PASS", "FAIL", "ABSTAIN"] = "PASS"
    error_code: Optional[str] = None
    message: str = "Audio passed validation checks"
    metadata: Optional[AudioMetadata] = None


def apply_dc_block(waveform: np.ndarray) -> np.ndarray:
    """
    Remove DC bias via mean-subtraction.
    Ensures zero mean before calculating energy and silence thresholds.
    """
    if waveform.size == 0:
        return waveform
    mean_val = float(np.mean(waveform))
    return waveform - mean_val


def compute_rms_dbfs(waveform: np.ndarray, eps: float = 1e-9) -> float:
    """Calculate the Root Mean Square level relative to full scale (dBFS)."""
    if waveform.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(waveform ** 2))
    return float(20.0 * np.log10(rms + eps))


def check_silence(waveform: np.ndarray, threshold_db: float = -60.0) -> Tuple[bool, float]:
    """
    Determine if the audio signal is functionally silent.
    Returns (is_silent, rms_dbfs).
    """
    dbfs = compute_rms_dbfs(waveform)
    is_silent = bool(dbfs < threshold_db)
    return is_silent, round(dbfs, 2)


def check_clipping(
    waveform: np.ndarray,
    peak_threshold: float = 0.99,
    min_consecutive_samples: int = 3,
    ratio_threshold: float = 0.001,
) -> Tuple[bool, float]:
    """
    Detect severe brickwall clipping by identifying runs of consecutive saturated samples.
    
    A single inter-sample peak at 0.99 may be transient, but flat-topped consecutive
    saturated samples indicate non-linear wave truncation that contorts harmonic content.
    
    Returns (is_clipped, clipping_ratio).
    """
    if waveform.size == 0:
        return False, 0.0

    saturated = np.abs(waveform) >= peak_threshold
    if not np.any(saturated):
        return False, 0.0

    # Identify consecutive runs using difference marking
    padded = np.concatenate(([0], saturated.astype(int), [0]))
    diff = np.diff(padded)
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]
    run_lengths = run_ends - run_starts

    # Count only runs satisfying minimum consecutive sample threshold (flat-topping)
    clipped_runs = run_lengths[run_lengths >= min_consecutive_samples]
    total_clipped_samples = int(np.sum(clipped_runs)) if clipped_runs.size > 0 else 0
    ratio = float(total_clipped_samples / waveform.size)

    is_clipped = bool(ratio > ratio_threshold)
    return is_clipped, round(ratio, 5)


def validate_format(file_path: str | Path, supported_formats: List[str]) -> ValidationResult:
    """Verify that file extension matches the allowlist."""
    path = Path(file_path)
    if not path.is_file():
        return ValidationResult(
            is_valid=False,
            status="FAIL",
            error_code="FILE_NOT_FOUND",
            message=f"Audio file does not exist: {path}",
        )

    ext = path.suffix.lower()
    allowed = [fmt.lower() for fmt in supported_formats]
    if ext not in allowed:
        return ValidationResult(
            is_valid=False,
            status="FAIL",
            error_code="UNSUPPORTED_FORMAT",
            message=f"File extension '{ext}' is not supported. Allowed: {allowed}",
        )

    return ValidationResult(is_valid=True, status="PASS", message="Format valid")


def validate_duration(duration_sec: float, min_duration: float, max_duration: float) -> ValidationResult:
    """Verify signal duration falls within stable feature extraction bounds."""
    if duration_sec < min_duration:
        return ValidationResult(
            is_valid=False,
            status="ABSTAIN",
            error_code="AUDIO_TOO_SHORT",
            message=f"Duration {duration_sec:.2f}s is below minimum {min_duration}s for stable analysis.",
        )
    if duration_sec > max_duration:
        return ValidationResult(
            is_valid=False,
            status="ABSTAIN",
            error_code="AUDIO_TOO_LONG",
            message=f"Duration {duration_sec:.2f}s exceeds maximum {max_duration}s limit.",
        )
    return ValidationResult(is_valid=True, status="PASS", message="Duration valid")

