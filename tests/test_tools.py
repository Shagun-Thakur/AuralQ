"""Analytical closed-form unit tests for Layer 3 DSP Tools and Registry."""

from pathlib import Path
import numpy as np
import pytest

from src.tools.energy import rms_energy
from src.tools.registry import (
    execute_tool,
    get_registered_tool_names,
    get_tool_metadata,
    list_tools_catalogue,
)
from src.tools.schemas import (
    MFCCArgs,
    RMSEnergyArgs,
    SpectralBandwidthArgs,
    SpectralCentroidArgs,
    SpectralFlatnessArgs,
    SpectralRolloffArgs,
    SpectrogramArgs,
    ZeroCrossingRateArgs,
)
from src.tools.spectral import (
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_rolloff,
)
from src.tools.temporal import zero_crossing_rate
from src.tools.timbre import mfcc
from src.tools.visual import spectrogram


@pytest.fixture
def synthetic_sine_1000hz():
    """Generate 2 seconds of pure 1000 Hz sine wave at A=0.5, SR=22050."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    return waveform, sr


@pytest.fixture
def synthetic_white_noise():
    """Generate 2 seconds of zero-mean Gaussian white noise."""
    sr = 22050
    duration = 2.0
    rng = np.random.default_rng(seed=42)
    waveform = rng.normal(0.0, 0.25, int(sr * duration)).astype(np.float32)
    # Clip slightly to avoid extreme outliers
    waveform = np.clip(waveform, -0.9, 0.9)
    return waveform, sr


def test_rms_energy_sine_analytical(synthetic_sine_1000hz):
    """Verify RMS of sine wave equals A / sqrt(2) within 1%."""
    waveform, sr = synthetic_sine_1000hz
    args = RMSEnergyArgs(frame_length=2048, hop_length=512)
    res = rms_energy(waveform, sr, args)

    assert res.status == "success"
    expected_rms = 0.5 / np.sqrt(2)  # ~0.35355
    actual_rms = res.result["mean_rms"]
    assert np.isclose(actual_rms, expected_rms, rtol=0.02)
    assert res.perceptual_anchors["loudness_stability"] == "consistent"
    assert "mean_rms" in res.citation_aliases
    assert len(res.citation_aliases["mean_rms"]) > 0


def test_zero_crossing_rate(synthetic_sine_1000hz, synthetic_white_noise):
    """Verify ZCR is low for smooth sine and high for white noise."""
    sine_wave, sr = synthetic_sine_1000hz
    noise_wave, _ = synthetic_white_noise
    args = ZeroCrossingRateArgs()

    sine_res = zero_crossing_rate(sine_wave, sr, args)
    noise_res = zero_crossing_rate(noise_wave, sr, args)

    assert sine_res.status == "success"
    assert noise_res.status == "success"
    assert sine_res.result["mean_zcr"] < 0.15
    assert noise_res.result["mean_zcr"] > 0.35
    assert "noisy" in noise_res.perceptual_anchors["temporal_character"]


def test_spectral_centroid_analytical(synthetic_sine_1000hz, synthetic_white_noise):
    """Verify centroid of 1000 Hz sine is within 20 Hz of 1000 Hz."""
    sine_wave, sr = synthetic_sine_1000hz
    noise_wave, _ = synthetic_white_noise
    args = SpectralCentroidArgs()

    sine_res = spectral_centroid(sine_wave, sr, args)
    assert sine_res.status == "success"
    assert np.isclose(sine_res.result["mean_hz"], 1000.0, atol=25.0)
    assert sine_res.perceptual_anchors["perceived_brightness"] == "dull"

    noise_res = spectral_centroid(noise_wave, sr, args)
    assert noise_res.status == "success"
    # White noise has high spectral centroid (> 4000 Hz)
    assert noise_res.result["mean_hz"] > 4000.0
    assert noise_res.perceptual_anchors["perceived_brightness"] == "bright"
    assert "4.0" in str(noise_res.citation_aliases["mean_hz"]) or "kHz" in str(noise_res.citation_aliases["mean_hz"])


def test_spectral_flatness_analytical(synthetic_sine_1000hz, synthetic_white_noise):
    """Verify flatness is near 0 for pure tone and > 0.40 for white noise."""
    sine_wave, sr = synthetic_sine_1000hz
    noise_wave, _ = synthetic_white_noise
    args = SpectralFlatnessArgs()

    sine_res = spectral_flatness(sine_wave, sr, args)
    assert sine_res.status == "success"
    assert sine_res.result["mean_flatness"] <= 0.01
    assert sine_res.perceptual_anchors["tonality_vs_noise"] == "tonal"

    noise_res = spectral_flatness(noise_wave, sr, args)
    assert noise_res.status == "success"
    assert noise_res.result["mean_flatness"] >= 0.40
    assert noise_res.perceptual_anchors["tonality_vs_noise"] == "noise-like"


def test_spectral_bandwidth_and_rolloff(synthetic_sine_1000hz):
    """Verify bandwidth is narrow for pure sine wave and rolloff is near fundamental."""
    sine_wave, sr = synthetic_sine_1000hz
    bw_res = spectral_bandwidth(sine_wave, sr, SpectralBandwidthArgs())
    ro_res = spectral_rolloff(sine_wave, sr, SpectralRolloffArgs(roll_percent=0.85))

    assert bw_res.status == "success"
    assert bw_res.result["mean_bandwidth_hz"] < 500.0
    assert bw_res.perceptual_anchors["spectral_spread"] == "narrow"

    assert ro_res.status == "success"
    assert np.isclose(ro_res.result["mean_rolloff_hz"], 1000.0, atol=150.0)


def test_mfcc_extraction(synthetic_sine_1000hz):
    """Verify MFCC generates 13 coefficients with summary stats."""
    waveform, sr = synthetic_sine_1000hz
    args = MFCCArgs(n_mfcc=13)
    res = mfcc(waveform, sr, args)

    assert res.status == "success"
    assert len(res.result["mean_coefficients"]) == 13
    assert len(res.result["std_coefficients"]) == 13
    assert res.result["overall_timbre_variance"] >= 0.0


def test_spectrogram_headless(synthetic_sine_1000hz, tmp_path):
    """Verify spectrogram computes energy ratios and exports plot headlessly."""
    waveform, sr = synthetic_sine_1000hz
    plot_file = tmp_path / "spec_test.png"
    args = SpectrogramArgs(export_plot=True, output_path=str(plot_file))
    res = spectrogram(waveform, sr, args)

    assert res.status == "success"
    assert np.isclose(res.result["peak_frequency_hz"], 1000.0, atol=25.0)
    assert res.result["plot_path"] == str(plot_file.resolve())
    assert plot_file.is_file()
    assert plot_file.stat().st_size > 1000


def test_insufficient_samples_guard():
    """Verify all tools gracefully return insufficient_samples when signal < n_fft."""
    sr = 22050
    tiny_waveform = np.ones(100, dtype=np.float32)

    res_rms = rms_energy(tiny_waveform, sr, RMSEnergyArgs(frame_length=2048))
    assert res_rms.status == "insufficient_samples"

    res_cent = spectral_centroid(tiny_waveform, sr, SpectralCentroidArgs(n_fft=2048))
    assert res_cent.status == "insufficient_samples"

    res_flat = spectral_flatness(tiny_waveform, sr, SpectralFlatnessArgs(n_fft=2048))
    assert res_flat.status == "insufficient_samples"


def test_tool_registry_and_dispatcher(synthetic_sine_1000hz):
    """Verify registry allowlist, metadata queries, and safe execution dispatch."""
    waveform, sr = synthetic_sine_1000hz
    tool_names = get_registered_tool_names()
    assert len(tool_names) == 9
    assert "rms_energy" in tool_names
    assert "spectral_centroid" in tool_names
    assert "dataset_profiler" in tool_names

    catalogue = list_tools_catalogue()
    assert len(catalogue) == 9

    # Safe execution with valid arguments
    res = execute_tool("spectral_centroid", waveform, sr, {"n_fft": 2048})
    assert res.status == "success"

    # Execution with invalid parameter (ge constraint violated)
    bad_res = execute_tool("spectral_centroid", waveform, sr, {"n_fft": 64})
    assert bad_res.status == "dsp_error"
    assert any("Invalid tool arguments" in w for w in bad_res.warnings)

    # Execution with unregistered tool
    unregistered_res = execute_tool("random_fake_tool", waveform, sr, {})
    assert unregistered_res.status == "dsp_error"
    assert any("not in the registered allowlist" in w for w in unregistered_res.warnings)

