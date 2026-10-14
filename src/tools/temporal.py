"""Temporal and transient analysis tools for AuralQ Layer 3."""

import time
import librosa
import numpy as np
from src.tools.schemas import ToolResultModel, ZeroCrossingRateArgs, generate_numeric_citation_aliases


def zero_crossing_rate(
    waveform: np.ndarray,
    sr: int,
    args: ZeroCrossingRateArgs,
) -> ToolResultModel:
    """
    Compute frame-by-frame Zero Crossing Rate (ZCR).
    Answers: 'Is this audio smooth, percussive, or rapidly varying?'
    """
    start_time = time.perf_counter()

    if waveform.size < args.frame_length:
        return ToolResultModel(
            tool="zero_crossing_rate",
            status="insufficient_samples",
            parameters=args.model_dump(),
            result={},
            perceptual_anchors={"temporal_character": "unknown (insufficient audio length)"},
            citation_aliases={},
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=[f"Signal size ({waveform.size}) is smaller than frame_length ({args.frame_length})."],
        )

    zcr = librosa.feature.zero_crossing_rate(
        y=waveform,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )[0]

    mean_zcr = float(np.mean(zcr))
    std_zcr = float(np.std(zcr))
    max_zcr = float(np.max(zcr))
    min_zcr = float(np.min(zcr))

    # Perceptual anchor mapping accounting for periodicity (std_zcr) and frequency/noise (mean_zcr)
    if std_zcr < 0.02:
        if mean_zcr < 0.12:
            anchor = "smooth / steady tonal pitch"
        else:
            anchor = "steady high-frequency tone"
    elif mean_zcr < 0.08:
        anchor = "smooth / low-frequency or voiced speech"
    elif mean_zcr <= 0.25:
        anchor = "moderate temporal variation / mixed audio"
    else:
        anchor = "noisy / percussive / rapidly varying (high transient content)"


    result_data = {
        "mean_zcr": round(mean_zcr, 4),
        "std_zcr": round(std_zcr, 4),
        "max_zcr": round(max_zcr, 4),
        "min_zcr": round(min_zcr, 4),
        "unit": "ratio",
    }

    aliases = {
        "mean_zcr": generate_numeric_citation_aliases(round(mean_zcr, 4)),
    }

    elapsed = (time.perf_counter() - start_time) * 1000

    return ToolResultModel(
        tool="zero_crossing_rate",
        status="success",
        parameters=args.model_dump(),
        result=result_data,
        perceptual_anchors={
            "temporal_character": anchor,
            "grounded_term": anchor,
        },
        citation_aliases=aliases,
        execution_time_ms=round(elapsed, 2),
    )

