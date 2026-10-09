"""Audio loading and preprocessing pipeline for AuralQ Layer 1."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field

from src.audio.dataset_scanner import scan_dataset_directory
from src.audio.validation import (
    AudioMetadata,
    ValidationResult,
    apply_dc_block,
    check_clipping,
    check_silence,
    validate_duration,
    validate_format,
)
from src.utils.config import get_audio_config


class AudioLoadResult(BaseModel):
    """Encapsulates the complete result of audio ingestion (single-file or dataset)."""
    status: str = "SUCCESS"  # SUCCESS, FAIL, ABSTAIN
    input_type: Literal["single_file", "dataset_directory"] = "single_file"
    error_code: Optional[str] = None
    message: str = "Audio loaded and preprocessed successfully"
    metadata: Optional[AudioMetadata] = None
    dataset_file_count: Optional[int] = None
    # Waveform is excluded from Pydantic serialization for lightweight transport
    waveform: Optional[Any] = Field(default=None, exclude=True)
    sample_rate: int = 22050

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_audio_file(
    file_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
) -> AudioLoadResult:
    """
    Ingest, format-validate, resample, DC-block, and quality-audit an audio file.
    
    Returns AudioLoadResult with status 'SUCCESS', 'FAIL', or 'ABSTAIN'.
    Never raises raw unhandled exceptions.
    """
    cfg = config or get_audio_config()
    ingest_cfg = cfg.get("ingestion", {})
    supported_formats = cfg.get("supported_formats", [".wav", ".mp3", ".flac", ".ogg"])

    target_sr = int(ingest_cfg.get("target_sr", 22050))
    to_mono = bool(ingest_cfg.get("mono", True))
    min_dur = float(ingest_cfg.get("min_duration_sec", 0.5))
    max_dur = float(ingest_cfg.get("max_duration_sec", 300.0))
    clip_thresh = float(ingest_cfg.get("clipping_threshold", 0.99))
    silence_thresh_db = float(ingest_cfg.get("silence_threshold_db", -60.0))

    path = Path(file_path)

    # 1. Format validation
    fmt_res = validate_format(path, supported_formats)
    if not fmt_res.is_valid:
        return AudioLoadResult(
            status=fmt_res.status,
            error_code=fmt_res.error_code,
            message=fmt_res.message,
        )

    # 2. Decode & Load
    try:
        y, orig_sr = librosa.load(str(path), sr=target_sr, mono=to_mono)
    except Exception as e:
        return AudioLoadResult(
            status="FAIL",
            error_code="DECODE_ERROR",
            message=f"Failed to decode audio file '{path.name}': {str(e)}",
        )

    # Convert to float32 contiguous array
    y = np.ascontiguousarray(y, dtype=np.float32)

    # 3. DC-block (mean subtraction)
    y = apply_dc_block(y)

    duration_sec = float(len(y) / target_sr) if target_sr > 0 else 0.0

    # 4. Duration validation
    dur_res = validate_duration(duration_sec, min_dur, max_dur)
    if not dur_res.is_valid:
        return AudioLoadResult(
            status=dur_res.status,
            error_code=dur_res.error_code,
            message=dur_res.message,
        )

    # 5. Silence detection
    is_silent, rms_dbfs = check_silence(y, threshold_db=silence_thresh_db)
    if is_silent:
        return AudioLoadResult(
            status="ABSTAIN",
            error_code="AUDIO_SILENT",
            message=f"Signal level ({rms_dbfs} dBFS) is below analytical silence threshold ({silence_thresh_db} dBFS).",
            metadata=AudioMetadata(
                file_path=str(path.resolve()),
                duration_sec=round(duration_sec, 3),
                sample_rate=target_sr,
                num_channels=1,
                num_samples=len(y),
                rms_dbfs=rms_dbfs,
            ),
        )

    # 6. Clipping detection
    is_clipped, clip_ratio = check_clipping(y, peak_threshold=clip_thresh)
    warnings: List[str] = []
    if is_clipped:
        warnings.append(
            f"CLIPPING_WARNING: {clip_ratio * 100:.2f}% of samples exhibit flat-topped clipping (>= {clip_thresh}). "
            "Harmonic and spectral measurements may contain distortion artifacts."
        )

    meta = AudioMetadata(
        file_path=str(path.resolve()),
        duration_sec=round(duration_sec, 3),
        sample_rate=target_sr,
        num_channels=1,
        num_samples=len(y),
        rms_dbfs=rms_dbfs,
        is_clipped=is_clipped,
        clipping_ratio=clip_ratio,
        warnings=warnings,
    )

    return AudioLoadResult(
        status="SUCCESS",
        input_type="single_file",
        metadata=meta,
        waveform=y,
        sample_rate=target_sr,
    )


def load_audio_input(
    input_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
) -> AudioLoadResult:
    """
    Unified dual-mode ingestion dispatcher.
    Accepts either an individual audio file OR an entire audio dataset directory.
    """
    path = Path(input_path)
    if not path.exists():
        return AudioLoadResult(
            status="FAIL",
            error_code="PATH_NOT_FOUND",
            message=f"Provided path does not exist: {path}",
        )

    if path.is_dir():
        files = scan_dataset_directory(path)
        if not files:
            return AudioLoadResult(
                status="FAIL",
                input_type="dataset_directory",
                error_code="EMPTY_DATASET_DIRECTORY",
                message=f"No supported audio files found in directory: {path}",
            )
        cfg = config or get_audio_config()
        target_sr = int(cfg.get("ingestion", {}).get("target_sr", 22050))
        meta = AudioMetadata(
            file_path=str(path.resolve()),
            duration_sec=0.0,
            sample_rate=target_sr,
            num_channels=0,
            num_samples=0,
            rms_dbfs=0.0,
            warnings=[f"DATASET_INPUT: {len(files)} audio files discovered"],
        )
        return AudioLoadResult(
            status="SUCCESS",
            input_type="dataset_directory",
            message=f"Discovered {len(files)} audio files in dataset directory.",
            dataset_file_count=len(files),
            metadata=meta,
            sample_rate=target_sr,
        )

    # Individual file mode
    return load_audio_file(path, config=config)

