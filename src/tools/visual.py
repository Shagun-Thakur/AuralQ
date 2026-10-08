"""Visualization and time-frequency summary tools for AuralQ Layer 3."""

import os
from pathlib import Path
import time
from typing import Optional
import librosa
import librosa.display
import numpy as np
from src.tools.schemas import SpectrogramArgs, ToolResultModel, generate_numeric_citation_aliases


def spectrogram(
    waveform: np.ndarray,
    sr: int,
    args: SpectrogramArgs,
) -> ToolResultModel:
    """
    Compute STFT magnitude spectrogram summary statistics and optional headless plot.
    Answers: 'How is frequency distributed over time?'
    """
    start_time = time.perf_counter()

    if waveform.size < args.n_fft:
        return ToolResultModel(
            tool="spectrogram",
            status="insufficient_samples",
            parameters=args.model_dump(),
            result={},
            perceptual_anchors={"spectral_balance": "unknown (insufficient audio length)"},
            citation_aliases={},
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=[f"Signal size ({waveform.size}) is smaller than n_fft ({args.n_fft})."],
        )

    # 1. Compute STFT magnitude
    stft = np.abs(librosa.stft(y=waveform, n_fft=args.n_fft, hop_length=args.hop_length))
    power = stft ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=args.n_fft)

    # 2. Extract frequency band energy distributions
    low_band_mask = freqs < 500.0
    mid_band_mask = (freqs >= 500.0) & (freqs <= 3000.0)
    high_band_mask = freqs > 3000.0

    total_energy = float(np.sum(power)) + 1e-9
    low_energy_ratio = float(np.sum(power[low_band_mask, :]) / total_energy)
    mid_energy_ratio = float(np.sum(power[mid_band_mask, :]) / total_energy)
    high_energy_ratio = float(np.sum(power[high_band_mask, :]) / total_energy)

    # Peak dominant frequency
    mean_power_per_bin = np.mean(power, axis=1)
    peak_bin = int(np.argmax(mean_power_per_bin))
    peak_freq_hz = float(freqs[peak_bin])

    # Perceptual anchor mapping
    if low_energy_ratio > 0.60:
        anchor = "bass / low-frequency dominated"
    elif high_energy_ratio > 0.40:
        anchor = "treble / high-frequency dominated"
    else:
        anchor = "mid-range balanced"

    result_data = {
        "peak_frequency_hz": round(peak_freq_hz, 2),
        "low_band_ratio": round(low_energy_ratio, 4),
        "mid_band_ratio": round(mid_energy_ratio, 4),
        "high_band_ratio": round(high_energy_ratio, 4),
        "plot_path": None,
    }

    # 3. Headless visual plot export if requested
    if args.export_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive, headless backend
            import matplotlib.pyplot as plt

            out_path = args.output_path or f"spectrogram_{int(time.time())}.png"
            fig, ax = plt.subplots(figsize=(8, 4))
            s_db = librosa.amplitude_to_db(stft, ref=np.max)
            img = librosa.display.specshow(
                s_db, sr=sr, hop_length=args.hop_length, x_axis="time", y_axis="hz", ax=ax
            )
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set(title="Spectrogram (STFT)")
            fig.tight_layout()

            # Ensure parent directories exist
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=120)
            plt.close(fig)  # Free memory immediately
            result_data["plot_path"] = str(Path(out_path).resolve())
        except Exception as e:
            result_data["plot_path"] = None

    aliases = {
        "peak_frequency_hz": generate_numeric_citation_aliases(round(peak_freq_hz, 2), unit="Hz"),
        "low_band_ratio": generate_numeric_citation_aliases(round(low_energy_ratio, 4)),
        "high_band_ratio": generate_numeric_citation_aliases(round(high_energy_ratio, 4)),
    }

    elapsed = (time.perf_counter() - start_time) * 1000

    return ToolResultModel(
        tool="spectrogram",
        status="success",
        parameters=args.model_dump(),
        result=result_data,
        perceptual_anchors={
            "spectral_balance": anchor,
            "grounded_term": anchor,
        },
        citation_aliases=aliases,
        execution_time_ms=round(elapsed, 2),
    )
