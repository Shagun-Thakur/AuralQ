"""Unit tests for Layer 1 Audio Ingestion and Hygiene using synthetic signals."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.audio.loader import load_audio_file
from src.audio.validation import (
    apply_dc_block,
    check_clipping,
    check_silence,
    compute_rms_dbfs,
    validate_duration,
    validate_format,
)


def test_dc_block_removes_bias():
    """Verify mean-subtraction completely zeroes out constant DC bias."""
    t = np.linspace(0, 1.0, 22050, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 1000 * t)
    biased_signal = sine + 0.35  # Inject 0.35 DC offset

    cleaned = apply_dc_block(biased_signal)
    assert np.isclose(np.mean(cleaned), 0.0, atol=1e-6)
    assert np.isclose(np.max(cleaned) - np.min(cleaned), 1.0, atol=1e-3)


def test_silence_detection():
    """Verify silence gating catches pure silence and near-silent signals."""
    sr = 22050
    zeros = np.zeros(sr, dtype=np.float32)
    is_silent, dbfs = check_silence(zeros, threshold_db=-60.0)
    assert is_silent is True
    assert dbfs < -60.0

    # 1 kHz sine at -20 dBFS amplitude (0.1)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    active_signal = 0.1 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    is_silent_active, dbfs_active = check_silence(active_signal, threshold_db=-60.0)
    assert is_silent_active is False
    assert dbfs_active > -30.0


def test_consecutive_clipping_detection():
    """Verify detection of flat-topped consecutive clipping runs vs transients."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Generate an overdriven sine wave clipped at 0.99
    raw_sine = 1.8 * np.sin(2 * np.pi * 200 * t)
    clipped_sine = np.clip(raw_sine, -0.99, 0.99).astype(np.float32)

    is_clipped, ratio = check_clipping(clipped_sine, peak_threshold=0.99, min_consecutive_samples=3)
    assert is_clipped is True
    assert ratio > 0.05  # Severe clipping (> 5% of samples)

    # Clean sine wave at 0.8 amplitude
    clean_sine = (0.8 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    clean_clipped, clean_ratio = check_clipping(clean_sine)
    assert clean_clipped is False
    assert clean_ratio == 0.0


def test_validate_duration():
    """Verify minimum and maximum duration constraints."""
    assert validate_duration(0.2, min_duration=0.5, max_duration=300.0).status == "ABSTAIN"
    assert validate_duration(350.0, min_duration=0.5, max_duration=300.0).status == "ABSTAIN"
    assert validate_duration(10.0, min_duration=0.5, max_duration=300.0).status == "PASS"


def test_validate_format(tmp_path):
    """Verify file extension allowlist enforcement."""
    valid_file = tmp_path / "sample.wav"
    valid_file.write_bytes(b"dummy")
    invalid_file = tmp_path / "sample.exe"
    invalid_file.write_bytes(b"dummy")

    assert validate_format(valid_file, [".wav", ".mp3"]).is_valid is True
    assert validate_format(invalid_file, [".wav", ".mp3"]).is_valid is False
    assert validate_format(tmp_path / "missing.wav", [".wav"]).is_valid is False


def test_load_audio_file_silence_abstention(tmp_path):
    """Verify that loading a silent audio file safely triggers ABSTAIN."""
    sr = 22050
    silent_audio = np.zeros(sr * 2, dtype=np.float32)
    silence_file = tmp_path / "silent.wav"
    sf.write(str(silence_file), silent_audio, sr)

    res = load_audio_file(silence_file)
    assert res.status == "ABSTAIN"
    assert res.error_code == "AUDIO_SILENT"
    assert res.metadata is not None
    assert res.metadata.rms_dbfs < -60.0


def test_load_audio_file_clipped_warning_injection(tmp_path):
    """Verify that clipped audio loads successfully but injects CLIPPING_WARNING."""
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    overdriven = np.clip(2.0 * np.sin(2 * np.pi * 440 * t), -0.99, 0.99).astype(np.float32)
    clipped_file = tmp_path / "clipped.wav"
    sf.write(str(clipped_file), overdriven, sr)

    res = load_audio_file(clipped_file)
    assert res.status == "SUCCESS"
    assert res.metadata is not None
    assert res.metadata.is_clipped is True
    assert any("CLIPPING_WARNING" in w for w in res.metadata.warnings)

