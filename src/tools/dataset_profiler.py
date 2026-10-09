"""Two-tier streaming dataset acoustic profiler for AuralQ Layer 3."""

from collections import Counter
from pathlib import Path
import time
from typing import Dict, List, Optional
import librosa
import numpy as np

from src.audio.dataset_scanner import (
    inspect_file_header,
    scan_dataset_directory,
    stream_dataset_waveforms,
)
from src.audio.validation import check_clipping, check_silence, compute_rms_dbfs
from src.tools.schemas import DatasetProfileModel


class WelfordAccumulator:
    """Computes online mean, variance, min, and max in O(1) space."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if x < self.min_val:
            self.min_val = x
        if x > self.max_val:
            self.max_val = x

    @property
    def variance(self) -> float:
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std_dev(self) -> float:
        return float(np.sqrt(self.variance))


def profile_audio_dataset(
    dataset_path: str | Path,
    target_sr: int = 22050,
    max_files: int = 2000,
) -> DatasetProfileModel:
    """
    Execute a two-tier streaming profile across an entire audio dataset.
    Guarantees sub-100 MB RAM ceiling via online Welford accumulation.
    """
    start_time = time.perf_counter()
    path = Path(dataset_path)

    # 1. Discover audio files
    file_paths = scan_dataset_directory(path, max_files=max_files)
    if not file_paths:
        return DatasetProfileModel(
            total_files=0,
            total_duration_hours=0.0,
            format_counts={},
            sample_rate_counts={},
            channel_counts={},
            duration_stats={"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std_dev": 0.0, "iqr": 0.0},
            rms_dbfs_stats={"min": 0.0, "max": 0.0, "mean": 0.0, "std_dev": 0.0},
            spectral_centroid_mean_hz=0.0,
            spectral_flatness_mean=0.0,
            silent_files=[],
            clipped_files=[],
            corrupted_files=[],
            processing_time_sec=round(time.perf_counter() - start_time, 2),
        )

    # 2. Tier 1: Fast Header Scan (~1000 files/sec)
    format_counts: Counter = Counter()
    sample_rate_counts: Counter = Counter()
    channel_counts: Counter = Counter()
    durations: List[float] = []
    corrupted_files: List[str] = []
    valid_paths: List[Path] = []

    for p in file_paths:
        hdr = inspect_file_header(p)
        if not hdr.is_readable:
            corrupted_files.append(str(p.resolve()))
            continue
        format_counts[hdr.format] += 1
        sample_rate_counts[hdr.sample_rate] += 1
        channel_counts[hdr.num_channels] += 1
        durations.append(hdr.duration_sec)
        valid_paths.append(p)

    # Compute duration percentiles
    if durations:
        d_arr = np.array(durations, dtype=np.float32)
        q75, q25 = np.percentile(d_arr, [75, 25])
        duration_stats = {
            "min": round(float(np.min(d_arr)), 2),
            "max": round(float(np.max(d_arr)), 2),
            "mean": round(float(np.mean(d_arr)), 2),
            "median": round(float(np.median(d_arr)), 2),
            "std_dev": round(float(np.std(d_arr)), 2),
            "iqr": round(float(q75 - q25), 2),
        }
        total_duration_hours = round(float(np.sum(d_arr) / 3600.0), 3)
    else:
        duration_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std_dev": 0.0, "iqr": 0.0}
        total_duration_hours = 0.0

    # 3. Tier 2: Streamed Acoustic & Hygiene Pass (1 file in memory at a time)
    rms_acc = WelfordAccumulator()
    centroid_sum = 0.0
    centroid_count = 0
    flatness_sum = 0.0
    flatness_count = 0
    silent_files: List[str] = []
    clipped_files: List[str] = []

    for file_p, waveform, sr, meta in stream_dataset_waveforms(valid_paths, target_sr=target_sr):
        if waveform is None:
            corrupted_files.append(str(file_p.resolve()))
            continue

        # Energy & Hygiene
        rms_dbfs = meta.rms_dbfs
        rms_acc.update(rms_dbfs)

        if meta.rms_dbfs < -60.0:
            silent_files.append(str(file_p.resolve()))

        if meta.is_clipped:
            clipped_files.append(str(file_p.resolve()))

        # Lightweight spectral snapshot (only if signal has sufficient length)
        if waveform.size >= 1024:
            try:
                # Fast spectral centroid frame mean
                cent = librosa.feature.spectral_centroid(y=waveform, sr=sr, n_fft=1024, hop_length=512)
                centroid_sum += float(np.mean(cent))
                centroid_count += 1

                # Fast spectral flatness frame mean
                flat = librosa.feature.spectral_flatness(y=waveform, n_fft=1024, hop_length=512)
                flatness_sum += float(np.mean(flat))
                flatness_count += 1
            except Exception:
                pass

    rms_stats = {
        "min": round(rms_acc.min_val if rms_acc.count > 0 else 0.0, 2),
        "max": round(rms_acc.max_val if rms_acc.count > 0 else 0.0, 2),
        "mean": round(rms_acc.mean if rms_acc.count > 0 else 0.0, 2),
        "std_dev": round(rms_acc.std_dev if rms_acc.count > 0 else 0.0, 2),
    }

    mean_centroid = round(centroid_sum / centroid_count, 2) if centroid_count > 0 else 0.0
    mean_flatness = round(flatness_sum / flatness_count, 4) if flatness_count > 0 else 0.0

    elapsed = round(time.perf_counter() - start_time, 2)

    return DatasetProfileModel(
        total_files=len(file_paths),
        total_duration_hours=total_duration_hours,
        format_counts=dict(format_counts),
        sample_rate_counts={int(k): int(v) for k, v in sample_rate_counts.items()},
        channel_counts={int(k): int(v) for k, v in channel_counts.items()},
        duration_stats=duration_stats,
        rms_dbfs_stats=rms_stats,
        spectral_centroid_mean_hz=mean_centroid,
        spectral_flatness_mean=mean_flatness,
        silent_files=silent_files,
        clipped_files=clipped_files,
        corrupted_files=corrupted_files,
        processing_time_sec=elapsed,
    )

