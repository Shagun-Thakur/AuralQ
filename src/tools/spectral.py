"""Spectral analysis tools for AuralQ Layer 3."""

import time
import librosa
import numpy as np
from src.tools.schemas import (
    SpectralBandwidthArgs,
    SpectralCentroidArgs,
    SpectralFlatnessArgs,
    SpectralRolloffArgs,
    ToolResultModel,
    generate_numeric_citation_aliases,
)


def spectral_centroid(
    waveform: np.ndarray,
    sr: int,
    args: SpectralCentroidArgs,
) -> ToolResultModel:
    """
    Compute spectral centroid (weighted mean frequency).
    Answers: 'Does this audio sound bright or dull?'
    """
    start_time = time.perf_counter()

    if waveform.size < args.n_fft:
        return ToolResultModel(
            tool="spectral_centroid",
            status="insufficient_samples",
            parameters=args.model_dump(),
            result={},
            perceptual_anchors={"perceived_brightness": "unknown (insufficient audio length)"},
            citation_aliases={},
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=[f"Signal size ({waveform.size}) is smaller than n_fft ({args.n_fft})."],
        )

    cent = librosa.feature.spectral_centroid(
        y=waveform,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )[0]

    mean_cent = float(np.mean(cent))
    std_cent = float(np.std(cent))
    median_cent = float(np.median(cent))

    # Perceptual anchor mapping
    if mean_cent >= 3500.0:
        anchor = "bright"
    elif mean_cent <= 1200.0:
        anchor = "dull"
    else:
        anchor = "balanced / neutral"

    result_data = {
        "mean_hz": round(mean_cent, 2),
        "std_hz": round(std_cent, 2),
        "median_hz": round(median_cent, 2),
        "unit": "Hz",
    }

    aliases = {
        "mean_hz": generate_numeric_citation_aliases(round(mean_cent, 2), unit="Hz"),
        "median_hz": generate_numeric_citation_aliases(round(median_cent, 2), unit="Hz"),
    }

    elapsed = (time.perf_counter() - start_time) * 1000

    return ToolResultModel(
        tool="spectral_centroid",
        status="success",
        parameters=args.model_dump(),
        result=result_data,
        perceptual_anchors={
            "perceived_brightness": anchor,
            "grounded_term": anchor,
        },
        citation_aliases=aliases,
        execution_time_ms=round(elapsed, 2),
    )


def spectral_bandwidth(
    waveform: np.ndarray,
    sr: int,
    args: SpectralBandwidthArgs,
) -> ToolResultModel:
    """
    Compute spectral bandwidth (spread of frequency content around centroid).
    Answers: 'Is the frequency content narrow or wide?'
    """
    start_time = time.perf_counter()

    if waveform.size < args.n_fft:
        return ToolResultModel(
            tool="spectral_bandwidth",
            status="insufficient_samples",
            parameters=args.model_dump(),
            result={},
            perceptual_anchors={"spectral_spread": "unknown (insufficient audio length)"},
            citation_aliases={},
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=[f"Signal size ({waveform.size}) is smaller than n_fft ({args.n_fft})."],
        )

    bw = librosa.feature.spectral_bandwidth(
        y=waveform,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        p=args.p,
    )[0]

    mean_bw = float(np.mean(bw))
    std_bw = float(np.std(bw))

    anchor = "narrow" if mean_bw < 1500.0 else "wide"

    result_data = {
        "mean_bandwidth_hz": round(mean_bw, 2),
        "std_bandwidth_hz": round(std_bw, 2),
        "unit": "Hz",
    }

    aliases = {
        "mean_bandwidth_hz": generate_numeric_citation_aliases(round(mean_bw, 2), unit="Hz"),
    }

    elapsed = (time.perf_counter() - start_time) * 1000

    return ToolResultModel(
        tool="spectral_bandwidth",
        status="success",
        parameters=args.model_dump(),
        result=result_data,
        perceptual_anchors={
            "spectral_spread": anchor,
            "grounded_term": anchor,
        },
        citation_aliases=aliases,
        execution_time_ms=round(elapsed, 2),
    )


def spectral_rolloff(
    waveform: np.ndarray,
    sr: int,
    args: SpectralRolloffArgs,
) -> ToolResultModel:
    """
    Compute spectral rolloff (frequency below which roll_percent of energy lies).
    Answers: 'Where is most of the audio energy concentrated?'
    """
    start_time = time.perf_counter()

    if waveform.size < args.n_fft:
        return ToolResultModel(
            tool="spectral_rolloff",
            status="insufficient_samples",
            parameters=args.model_dump(),
            result={},
            perceptual_anchors={"energy_cutoff": "unknown (insufficient audio length)"},
            citation_aliases={},
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=[f"Signal size ({waveform.size}) is smaller than n_fft ({args.n_fft})."],
        )

    rolloff = librosa.feature.spectral_rolloff(
        y=waveform,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        roll_percent=args.roll_percent,
    )[0]

    mean_ro = float(np.mean(rolloff))
    std_ro = float(np.std(rolloff))

    anchor = "low-frequency concentrated" if mean_ro < 3000.0 else "high-frequency extended"

    result_data = {
        "mean_rolloff_hz": round(mean_ro, 2),
        "std_rolloff_hz": round(std_ro, 2),
        "roll_percent": args.roll_percent,
        "unit": "Hz",
    }

    aliases = {
        "mean_rolloff_hz": generate_numeric_citation_aliases(round(mean_ro, 2), unit="Hz"),
    }

    elapsed = (time.perf_counter() - start_time) * 1000

    return ToolResultModel(
        tool="spectral_rolloff",
        status="success",
        parameters=args.model_dump(),
        result=result_data,
        perceptual_anchors={
            "energy_cutoff": anchor,
            "grounded_term": anchor,
        },
        citation_aliases=aliases,
        execution_time_ms=round(elapsed, 2),
    )


def spectral_flatness(
    waveform: np.ndarray,
    sr: int,
    args: SpectralFlatnessArgs,
) -> ToolResultModel:
    """
    Compute spectral flatness (ratio of geometric to arithmetic mean of power spectrum).
    Answers: 'Does this sound noise-like or tonal?'
    """
    start_time = time.perf_counter()

    if waveform.size < args.n_fft:
        return ToolResultModel(
            tool="spectral_flatness",
            status="insufficient_samples",
            parameters=args.model_dump(),
            result={},
            perceptual_anchors={"tonality_vs_noise": "unknown (insufficient audio length)"},
            citation_aliases={},
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=[f"Signal size ({waveform.size}) is smaller than n_fft ({args.n_fft})."],
        )

    flatness = librosa.feature.spectral_flatness(
        y=waveform,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )[0]

    mean_flat = float(np.mean(flatness))
    std_flat = float(np.std(flatness))

    # Perceptual anchor mapping
    if mean_flat <= 0.05:
        anchor = "tonal"
    elif mean_flat <= 0.25:
        anchor = "moderately tonal"
    else:
        anchor = "noise-like"

    result_data = {
        "mean_flatness": round(mean_flat, 4),
        "std_flatness": round(std_flat, 4),
        "unit": "ratio_0_to_1",
    }

    aliases = {
        "mean_flatness": generate_numeric_citation_aliases(round(mean_flat, 4)),
    }

    elapsed = (time.perf_counter() - start_time) * 1000

    return ToolResultModel(
        tool="spectral_flatness",
        status="success",
        parameters=args.model_dump(),
        result=result_data,
        perceptual_anchors={
            "tonality_vs_noise": anchor,
            "grounded_term": anchor,
        },
        citation_aliases=aliases,
        execution_time_ms=round(elapsed, 2),
    )

