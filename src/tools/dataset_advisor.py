"""Deterministic expert rule engine for audio dataset preprocessing recommendations."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from src.tools.dataset_profiler import profile_audio_dataset
from src.tools.schemas import (
    DatasetAdvisorResultModel,
    DatasetProfileArgs,
    DatasetProfileModel,
    PreprocessingAdviceModel,
    ToolResultModel,
    generate_numeric_citation_aliases,
)


def generate_preprocessing_recommendations(
    profile: DatasetProfileModel,
) -> List[PreprocessingAdviceModel]:
    """
    Evaluate deterministic heuristic matrix over dataset distribution metrics.
    Every recommendation is mathematically grounded in measured thresholds.
    """
    recommendations: List[PreprocessingAdviceModel] = []
    total = max(profile.total_files, 1)

    # 1. Sample Rate Consistency Rule
    if len(profile.sample_rate_counts) > 1:
        modal_sr = max(profile.sample_rate_counts, key=profile.sample_rate_counts.get)
        divergent_count = sum(c for sr, c in profile.sample_rate_counts.items() if sr != modal_sr)
        pct = round((divergent_count / total) * 100, 1)
        sr_list = ", ".join(f"{sr} Hz ({c} files)" for sr, c in sorted(profile.sample_rate_counts.items()))
        recommendations.append(
            PreprocessingAdviceModel(
                action="RESAMPLE",
                priority="CRITICAL",
                reason=f"Multiple sample rates detected: {sr_list}. Inconsistent sample rates will cause model input shape mismatches or pitch distortion.",
                affected_files_count=divergent_count,
                affected_files_percentage=pct,
                parameters={"target_sample_rate": modal_sr, "recommended_method": "kaiser_best / soxr"},
                canonical_citations=[f"{modal_sr} Hz", f"{pct}%", str(divergent_count)],
            )
        )

    # 2. Multi-Channel Layout Rule
    stereo_or_multi = sum(c for ch, c in profile.channel_counts.items() if ch > 1)
    if stereo_or_multi > 0:
        pct = round((stereo_or_multi / total) * 100, 1)
        recommendations.append(
            PreprocessingAdviceModel(
                action="MONO_DOWNMIX",
                priority="RECOMMENDED",
                reason=f"{stereo_or_multi} files ({pct}%) have stereo or multi-channel audio. Downmixing to mono standardizes acoustic representation and halves memory usage.",
                affected_files_count=stereo_or_multi,
                affected_files_percentage=pct,
                parameters={"method": "equal_power_summing", "formula": "0.707 * (L + R)"},
                canonical_citations=[f"{pct}%", str(stereo_or_multi)],
            )
        )

    # 3. Duration Variance Rule (Chunking or Padding)
    d_stats = profile.duration_stats
    med_dur = d_stats.get("median", 0.0)
    dur_range = d_stats.get("max", 0.0) - d_stats.get("min", 0.0)
    iqr = d_stats.get("iqr", 0.0)
    if med_dur > 0 and ((dur_range / med_dur) > 0.5 or iqr > 3.0):
        recommendations.append(
            PreprocessingAdviceModel(
                action="CHUNKING_OR_PADDING",
                priority="RECOMMENDED",
                reason=f"Significant duration skew detected: min {d_stats.get('min')}s, max {d_stats.get('max')}s, IQR {iqr}s. Variable-length sequences waste computation on tensor zero-padding.",
                affected_files_count=total,
                affected_files_percentage=100.0,
                parameters={"recommended_window_sec": med_dur, "padding_mode": "constant_zeros_or_wrap"},
                canonical_citations=[f"{d_stats.get('min')}s", f"{d_stats.get('max')}s", f"{iqr}s"],
            )
        )

    # 4. Loudness Normalization Rule
    rms_stats = profile.rms_dbfs_stats
    std_rms = rms_stats.get("std_dev", 0.0)
    mean_rms = rms_stats.get("mean", 0.0)
    if std_rms > 6.0 or mean_rms < -35.0:
        recommendations.append(
            PreprocessingAdviceModel(
                action="LOUDNESS_NORMALIZATION",
                priority="RECOMMENDED",
                reason=f"Wide loudness discrepancy across dataset (mean: {mean_rms} dBFS, standard deviation: {std_rms} dB). Large gain variance destabilizes neural loss optimization.",
                affected_files_count=total,
                affected_files_percentage=100.0,
                parameters={"target_level_dbfs": -20.0, "algorithm": "EBU_R128_or_peak_norm", "headroom_db": 1.0},
                canonical_citations=[f"{std_rms} dB", f"{mean_rms} dBFS"],
            )
        )

    # 5. Silent File Pruning Rule
    silent_count = len(profile.silent_files)
    if silent_count > 0:
        pct = round((silent_count / total) * 100, 1)
        recommendations.append(
            PreprocessingAdviceModel(
                action="PRUNE_SILENCE",
                priority="CRITICAL",
                reason=f"{silent_count} files ({pct}%) are functionally silent (< -60 dBFS). Silent recordings cause vanishing gradients and invalid division in spectral tools.",
                affected_files_count=silent_count,
                affected_files_percentage=pct,
                parameters={"threshold_dbfs": -60.0, "file_paths": profile.silent_files},
                canonical_citations=[str(silent_count), f"{pct}%"],
            )
        )

    # 6. Clipped File Declipping or Removal Rule
    clipped_count = len(profile.clipped_files)
    if clipped_count > 0:
        pct = round((clipped_count / total) * 100, 1)
        recommendations.append(
            PreprocessingAdviceModel(
                action="PRUNE_OR_DECLIP",
                priority="RECOMMENDED",
                reason=f"{clipped_count} files ({pct}%) exhibit severe flat-topped clipping. Harmonic and rolloff measurements will be contaminated by non-linear distortion.",
                affected_files_count=clipped_count,
                affected_files_percentage=pct,
                parameters={"threshold_peak": 0.99, "remediation": "cubic_spline_declip_or_discard"},
                canonical_citations=[str(clipped_count), f"{pct}%"],
            )
        )

    # 7. Corrupted File Alert Rule
    corrupted_count = len(profile.corrupted_files)
    if corrupted_count > 0:
        pct = round((corrupted_count / total) * 100, 1)
        recommendations.append(
            PreprocessingAdviceModel(
                action="PURGE_CORRUPTED_FILES",
                priority="CRITICAL",
                reason=f"{corrupted_count} files ({pct}%) failed header decode. These unreadable files must be removed prior to pipeline execution.",
                affected_files_count=corrupted_count,
                affected_files_percentage=pct,
                parameters={"corrupted_paths": profile.corrupted_files},
                canonical_citations=[str(corrupted_count), f"{pct}%"],
            )
        )

    # 8. High-Pass Filter (Rumble / DC Bias) Rule
    if profile.spectral_centroid_mean_hz > 0 and profile.spectral_centroid_mean_hz < 800.0:
        recommendations.append(
            PreprocessingAdviceModel(
                action="HIGH_PASS_FILTER",
                priority="OPTIONAL",
                reason=f"Dataset is heavily low-frequency dominant (mean centroid: {profile.spectral_centroid_mean_hz} Hz). A high-pass filter removes sub-audible rumble.",
                affected_files_count=total,
                affected_files_percentage=100.0,
                parameters={"filter_type": "butterworth_highpass", "cutoff_hz": 60, "order": 2},
                canonical_citations=[f"{profile.spectral_centroid_mean_hz} Hz"],
            )
        )

    return recommendations


def dataset_profiler_tool(
    dataset_path: str,
    target_sr: int = 22050,
    max_files: int = 2000,
) -> ToolResultModel:
    """
    Layer 3 DSP Tool wrapper conforming to universal ToolResultModel interface.
    Computes two-tier acoustic profiling and deterministic preprocessing advice.
    """
    profile = profile_audio_dataset(dataset_path=dataset_path, target_sr=target_sr, max_files=max_files)
    
    if profile.total_files == 0:
        return ToolResultModel(
            tool="dataset_profiler",
            status="insufficient_samples",
            parameters={"dataset_path": dataset_path, "target_sr": target_sr, "max_files": max_files},
            result={},
            perceptual_anchors={"dataset_status": "empty or directory not found"},
            citation_aliases={},
            execution_time_ms=profile.processing_time_sec * 1000,
            warnings=[f"No valid audio files found in directory: '{dataset_path}'"],
        )

    recommendations = generate_preprocessing_recommendations(profile)

    # Build executive plain-English summary
    crit_actions = [r.action for r in recommendations if r.priority == "CRITICAL"]
    rec_actions = [r.action for r in recommendations if r.priority == "RECOMMENDED"]

    summary_parts = [
        f"Analyzed {profile.total_files} audio files totaling {profile.total_duration_hours:.2f} hours.",
    ]
    if crit_actions:
        summary_parts.append(f"CRITICAL actions required: {', '.join(crit_actions)}.")
    if rec_actions:
        summary_parts.append(f"Recommended optimizations: {', '.join(rec_actions)}.")
    if not crit_actions and not rec_actions:
        summary_parts.append("Dataset exhibits uniform acoustic hygiene; ready for direct feature extraction.")

    executive_summary = " ".join(summary_parts)

    # Pack into universal ToolResultModel
    result_data = {
        "dataset_path": dataset_path,
        "total_files": profile.total_files,
        "total_duration_hours": profile.total_duration_hours,
        "format_distribution": profile.format_counts,
        "sample_rate_distribution": profile.sample_rate_counts,
        "channel_distribution": profile.channel_counts,
        "duration_statistics": profile.duration_stats,
        "loudness_rms_dbfs": profile.rms_dbfs_stats,
        "spectral_centroid_mean_hz": profile.spectral_centroid_mean_hz,
        "spectral_flatness_mean": profile.spectral_flatness_mean,
        "silent_files_count": len(profile.silent_files),
        "clipped_files_count": len(profile.clipped_files),
        "corrupted_files_count": len(profile.corrupted_files),
        "recommendations": [r.model_dump() for r in recommendations],
        "executive_summary": executive_summary,
    }

    # Generate citation aliases
    citation_aliases = {
        "total_files": [str(profile.total_files)],
        "duration_hours": generate_numeric_citation_aliases(profile.total_duration_hours, unit="hours"),
        "mean_centroid": generate_numeric_citation_aliases(profile.spectral_centroid_mean_hz, unit="Hz"),
    }

    return ToolResultModel(
        tool="dataset_profiler",
        status="success",
        parameters={"dataset_path": dataset_path, "target_sr": target_sr, "max_files": max_files},
        result=result_data,
        perceptual_anchors={
            "dataset_hygiene": "action_required" if crit_actions else "sound",
            "loudness_profile": "variable" if profile.rms_dbfs_stats.get("std_dev", 0.0) > 6.0 else "stable",
        },
        citation_aliases=citation_aliases,
        execution_time_ms=profile.processing_time_sec * 1000,
    )

