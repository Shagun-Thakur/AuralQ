"""Energy and loudness analysis tools for AuralQ Layer 3."""

import time
import librosa
import numpy as np
from src.tools.schemas import RMSEnergyArgs, ToolResultModel, generate_numeric_citation_aliases


def rms_energy(
    waveform: np.ndarray,
    sr: int,
    args: RMSEnergyArgs,
) -> ToolResultModel:
    """
    Compute frame-by-frame RMS energy and stability statistics.
    Answers: 'Is this recording consistently loud or fluctuating?'
    """
    start_time = time.perf_counter()

    if waveform.size < args.frame_length:
        return ToolResultModel(
            tool="rms_energy",
            status="insufficient_samples",
            parameters=args.model_dump(),
            result={},
            perceptual_anchors={"loudness_character": "unknown (insufficient audio length)"},
            citation_aliases={},
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=[f"Signal size ({waveform.size}) is smaller than frame_length ({args.frame_length})."],
        )

    rms = librosa.feature.rms(
        y=waveform,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )[0]

    mean_rms = float(np.mean(rms))
    std_rms = float(np.std(rms))
    max_rms = float(np.max(rms))
    min_rms = float(np.min(rms))

    # Coefficient of variation (CV = std / mean) describes loudness stability
    cv = float(std_rms / (mean_rms + 1e-9))

    # Perceptual stability anchor mapping
    if cv < 0.15:
        stability_anchor = "consistent"
    elif cv < 0.35:
        stability_anchor = "moderately fluctuating"
    else:
        stability_anchor = "highly fluctuating / dynamic"

    # Convert linear RMS mean to dBFS for absolute perceptual volume level
    mean_rms_dbfs = float(20.0 * np.log10(mean_rms + 1e-9))

    # Perceptual level anchor mapping
    if mean_rms_dbfs < -45.0:
        level_anchor = "quiet / faint"
    elif mean_rms_dbfs < -22.0:
        level_anchor = "moderate volume"
    elif mean_rms_dbfs < -8.0:
        level_anchor = "loud / high energy"
    else:
        level_anchor = "very loud / near full-scale ceiling"

    result_data = {
        "mean_rms": round(mean_rms, 4),
        "mean_rms_dbfs": round(mean_rms_dbfs, 2),
        "std_rms": round(std_rms, 4),
        "max_rms": round(max_rms, 4),
        "min_rms": round(min_rms, 4),
        "dynamic_range_cv": round(cv, 4),
        "unit": "linear_amplitude_and_dbfs",
    }

    aliases = {
        "mean_rms": generate_numeric_citation_aliases(round(mean_rms, 4)),
        "mean_rms_dbfs": generate_numeric_citation_aliases(round(mean_rms_dbfs, 2), unit="dBFS"),
        "dynamic_range_cv": generate_numeric_citation_aliases(round(cv, 4)),
    }

    elapsed = (time.perf_counter() - start_time) * 1000

    return ToolResultModel(
        tool="rms_energy",
        status="success",
        parameters=args.model_dump(),
        result=result_data,
        perceptual_anchors={
            "loudness_stability": stability_anchor,
            "perceived_level": level_anchor,
            "grounded_term": f"{level_anchor}, {stability_anchor}",
        },
        citation_aliases=aliases,
        execution_time_ms=round(elapsed, 2),
    )


