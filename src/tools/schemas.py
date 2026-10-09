"""Pydantic v2 schemas and validation models for Layer 3 DSP Tools."""

import math
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class BaseToolArgs(BaseModel):
    """Base arguments model with strict validation configuration."""
    model_config = {"extra": "forbid"}


class RMSEnergyArgs(BaseToolArgs):
    """Arguments for RMS energy calculation."""
    frame_length: int = Field(default=2048, ge=256, le=8192)
    hop_length: int = Field(default=512, ge=64, le=4096)


class ZeroCrossingRateArgs(BaseToolArgs):
    """Arguments for Zero Crossing Rate calculation."""
    frame_length: int = Field(default=2048, ge=256, le=8192)
    hop_length: int = Field(default=512, ge=64, le=4096)


class SpectralCentroidArgs(BaseToolArgs):
    """Arguments for Spectral Centroid calculation."""
    n_fft: int = Field(default=2048, ge=512, le=8192)
    hop_length: int = Field(default=512, ge=128, le=2048)


class SpectralBandwidthArgs(BaseToolArgs):
    """Arguments for Spectral Bandwidth calculation."""
    n_fft: int = Field(default=2048, ge=512, le=8192)
    hop_length: int = Field(default=512, ge=128, le=2048)
    p: float = Field(default=2.0, ge=1.0, le=4.0)


class SpectralRolloffArgs(BaseToolArgs):
    """Arguments for Spectral Rolloff calculation."""
    n_fft: int = Field(default=2048, ge=512, le=8192)
    hop_length: int = Field(default=512, ge=128, le=2048)
    roll_percent: float = Field(default=0.85, ge=0.50, le=0.99)


class SpectralFlatnessArgs(BaseToolArgs):
    """Arguments for Spectral Flatness calculation."""
    n_fft: int = Field(default=2048, ge=512, le=8192)
    hop_length: int = Field(default=512, ge=128, le=2048)


class MFCCArgs(BaseToolArgs):
    """Arguments for Mel-Frequency Cepstral Coefficients calculation."""
    n_mfcc: int = Field(default=13, ge=1, le=40)
    n_fft: int = Field(default=2048, ge=512, le=8192)
    hop_length: int = Field(default=512, ge=128, le=2048)


class SpectrogramArgs(BaseToolArgs):
    """Arguments for Spectrogram summary and optional visual export."""
    n_fft: int = Field(default=2048, ge=512, le=8192)
    hop_length: int = Field(default=512, ge=128, le=2048)
    export_plot: bool = Field(default=False)
    output_path: Optional[str] = Field(default=None)


class DatasetProfileArgs(BaseToolArgs):
    """Arguments for dataset profiling and preprocessing recommendation."""
    dataset_path: str = Field(description="Directory path containing the audio dataset.")
    target_sr: Optional[int] = Field(default=22050, ge=8000, le=48000)
    max_files: int = Field(default=2000, ge=1, le=5000)


class PreprocessingAdviceModel(BaseModel):
    """Structured, evidence-grounded recommendation for dataset preprocessing."""
    action: str
    priority: Literal["CRITICAL", "RECOMMENDED", "OPTIONAL"]
    reason: str
    affected_files_count: int
    affected_files_percentage: float
    parameters: Dict[str, Any] = Field(default_factory=dict)
    canonical_citations: List[str] = Field(default_factory=list)


class DatasetProfileModel(BaseModel):
    """Comprehensive statistical distribution metrics for an audio corpus."""
    total_files: int
    total_duration_hours: float
    format_counts: Dict[str, int]
    sample_rate_counts: Dict[int, int]
    channel_counts: Dict[int, int]
    duration_stats: Dict[str, float]
    rms_dbfs_stats: Dict[str, float]
    spectral_centroid_mean_hz: float
    spectral_flatness_mean: float
    silent_files: List[str] = Field(default_factory=list)
    clipped_files: List[str] = Field(default_factory=list)
    corrupted_files: List[str] = Field(default_factory=list)
    processing_time_sec: float


class DatasetAdvisorResultModel(BaseModel):
    """Encapsulates the complete dataset profile, anomaly audits, and advice."""
    dataset_path: str
    profile: DatasetProfileModel
    recommendations: List[PreprocessingAdviceModel]
    executive_summary: str
    status: Literal["success", "empty_dataset", "error"]
    warnings: List[str] = Field(default_factory=list)


class ToolResultModel(BaseModel):
    """Universal structured result contract for all AuralQ DSP tools."""
    tool: str
    status: Literal["success", "insufficient_samples", "dsp_error"]
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    perceptual_anchors: Dict[str, str] = Field(default_factory=dict)
    citation_aliases: Dict[str, List[str]] = Field(default_factory=dict)
    execution_time_ms: float
    warnings: List[str] = Field(default_factory=list)


def generate_numeric_citation_aliases(
    val: float,
    unit: Optional[str] = None,
    round_decimals: int = 2,
) -> List[str]:
    """
    Generate canonical string tokens for a numeric measurement to make downstream
    regex citation validators resilient against natural engineering notations.
    """
    if math.isnan(val) or math.isinf(val):
        return []

    aliases: List[str] = []

    # Integer and standard decimal string representations
    int_str = str(int(round(val)))
    dec_full = f"{val:.{round_decimals}f}"
    dec_single = f"{val:.1f}"

    aliases.extend([dec_full, dec_single, int_str])

    # Suffix with unit if provided
    if unit:
        u_clean = unit.strip()
        aliases.extend([
            f"{dec_full} {u_clean}",
            f"{dec_single} {u_clean}",
            f"{int_str} {u_clean}",
            f"{dec_full}{u_clean}",
            f"{int_str}{u_clean}",
        ])
        # Hz to kHz conversion support
        if u_clean.lower() == "hz" and val >= 1000.0:
            khz_val = val / 1000.0
            aliases.extend([
                f"{khz_val:.1f} kHz",
                f"{khz_val:.2f} kHz",
                f"{khz_val:.1f}kHz",
                f"{khz_val:.2f}kHz",
            ])

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    return deduped

