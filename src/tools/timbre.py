"""Timbre and spectral envelope analysis tools for AuralQ Layer 3."""

import time
import librosa
import numpy as np
from src.tools.schemas import MFCCArgs, ToolResultModel, generate_numeric_citation_aliases


def mfcc(
    waveform: np.ndarray,
    sr: int,
    args: MFCCArgs,
) -> ToolResultModel:
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCCs).
    Answers: 'What is the timbral color or texture of this sound?'
    """
    start_time = time.perf_counter()

    if waveform.size < args.n_fft:
        return ToolResultModel(
            tool="mfcc",
            status="insufficient_samples",
            parameters=args.model_dump(),
            result={},
            perceptual_anchors={"timbral_texture": "unknown (insufficient audio length)"},
            citation_aliases={},
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=[f"Signal size ({waveform.size}) is smaller than n_fft ({args.n_fft})."],
        )

    mfccs = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    means = [round(float(m), 4) for m in np.mean(mfccs, axis=1)]
    stds = [round(float(s), 4) for s in np.std(mfccs, axis=1)]

    # Overall timbral variance (mean of standard deviations across coefficients)
    timbre_variance = float(np.mean(stds))
    if timbre_variance < 5.0:
        anchor = "static timbre"
    elif timbre_variance < 15.0:
        anchor = "moderately dynamic timbre"
    else:
        anchor = "highly evolving / heterogeneous timbre"

    result_data = {
        "n_mfcc": args.n_mfcc,
        "mean_coefficients": means,
        "std_coefficients": stds,
        "overall_timbre_variance": round(timbre_variance, 4),
        "unit": "cepstral_coefficients",
    }

    aliases = {
        "overall_timbre_variance": generate_numeric_citation_aliases(round(timbre_variance, 4)),
    }

    elapsed = (time.perf_counter() - start_time) * 1000

    return ToolResultModel(
        tool="mfcc",
        status="success",
        parameters=args.model_dump(),
        result=result_data,
        perceptual_anchors={
            "timbral_texture": anchor,
            "grounded_term": anchor,
        },
        citation_aliases=aliases,
        execution_time_ms=round(elapsed, 2),
    )

