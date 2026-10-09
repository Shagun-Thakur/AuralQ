"""Unit and analytical tests for Dataset Ingestion, Profiling, and Preprocessing Advisor."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.audio.loader import load_audio_input
from src.tools.dataset_advisor import dataset_profiler_tool, generate_preprocessing_recommendations
from src.tools.dataset_profiler import profile_audio_dataset
from src.tools.registry import execute_tool, get_registered_tool_names


@pytest.fixture
def synthetic_dataset_dir(tmp_path):
    """
    Create a controlled synthetic dataset fixture containing:
    1. Mixed sample rates (16 kHz and 22.05 kHz)
    2. Multi-channel audio (stereo file)
    3. Large duration disparity (1.0s to 6.0s)
    4. One completely silent file
    5. One severely clipped file
    6. One clean mono reference file
    """
    dataset_dir = tmp_path / "audio_corpus"
    dataset_dir.mkdir()

    # 1. 16 kHz Mono Sine
    sr_16k = 16000
    t_1s = np.linspace(0, 1.0, sr_16k, endpoint=False)
    y_16k = (0.5 * np.sin(2 * np.pi * 440 * t_1s)).astype(np.float32)
    sf.write(str(dataset_dir / "f1_16k_mono.wav"), y_16k, sr_16k)

    # 2. 22.05 kHz Stereo Sine
    sr_22k = 22050
    t_1_5s = np.linspace(0, 1.5, int(sr_22k * 1.5), endpoint=False)
    ch1 = (0.4 * np.sin(2 * np.pi * 880 * t_1_5s)).astype(np.float32)
    ch2 = (0.4 * np.sin(2 * np.pi * 440 * t_1_5s)).astype(np.float32)
    y_stereo = np.column_stack([ch1, ch2])
    sf.write(str(dataset_dir / "f2_22k_stereo.wav"), y_stereo, sr_22k)

    # 3. 22.05 kHz Long Mono Sine (6.0s)
    t_6s = np.linspace(0, 6.0, int(sr_22k * 6.0), endpoint=False)
    y_long = (0.3 * np.sin(2 * np.pi * 1000 * t_6s)).astype(np.float32)
    sf.write(str(dataset_dir / "f3_22k_long.wav"), y_long, sr_22k)

    # 4. Pure Silence File (< -60 dBFS)
    y_silent = np.zeros(int(sr_22k * 1.0), dtype=np.float32)
    sf.write(str(dataset_dir / "f4_silent.wav"), y_silent, sr_22k)

    # 5. Severely Clipped Sine File
    t_2s = np.linspace(0, 2.0, int(sr_22k * 2.0), endpoint=False)
    y_clipped = np.clip(2.5 * np.sin(2 * np.pi * 300 * t_2s), -0.99, 0.99).astype(np.float32)
    sf.write(str(dataset_dir / "f5_clipped.wav"), y_clipped, sr_22k)

    # 6. Standard Clean Reference Mono File (2.0s)
    y_clean = (0.4 * np.sin(2 * np.pi * 1000 * t_2s)).astype(np.float32)
    sf.write(str(dataset_dir / "f6_clean.wav"), y_clean, sr_22k)

    return dataset_dir


def test_dual_mode_loader_distinction(synthetic_dataset_dir):
    """Verify that load_audio_input cleanly differentiates single file vs directory."""
    # Test directory detection
    res_dir = load_audio_input(synthetic_dataset_dir)
    assert res_dir.status == "SUCCESS"
    assert res_dir.input_type == "dataset_directory"
    assert res_dir.dataset_file_count == 6

    # Test single file detection
    single_file = synthetic_dataset_dir / "f6_clean.wav"
    res_file = load_audio_input(single_file)
    assert res_file.status == "SUCCESS"
    assert res_file.input_type == "single_file"
    assert res_file.waveform is not None
    assert res_file.sample_rate == 22050


def test_streaming_dataset_profiler_metrics(synthetic_dataset_dir):
    """Verify statistical distributions and hygiene anomaly detection."""
    profile = profile_audio_dataset(synthetic_dataset_dir)

    assert profile.total_files == 6
    assert 16000 in profile.sample_rate_counts
    assert 22050 in profile.sample_rate_counts
    assert profile.channel_counts[1] >= 4
    assert profile.channel_counts[2] == 1  # Exactly 1 stereo file

    # Duration stats check
    assert profile.duration_stats["min"] == 1.0
    assert profile.duration_stats["max"] == 6.0
    assert profile.duration_stats["median"] >= 1.5

    # Anomalies detection
    assert len(profile.silent_files) == 1
    assert "f4_silent.wav" in profile.silent_files[0]

    assert len(profile.clipped_files) == 1
    assert "f5_clipped.wav" in profile.clipped_files[0]

    assert len(profile.corrupted_files) == 0


def test_deterministic_preprocessing_advisor(synthetic_dataset_dir):
    """Verify that the expert heuristic matrix outputs mathematically grounded advice."""
    profile = profile_audio_dataset(synthetic_dataset_dir)
    recommendations = generate_preprocessing_recommendations(profile)

    actions = {r.action: r for r in recommendations}

    # 1. RESAMPLE must be triggered (16k and 22k mixed)
    assert "RESAMPLE" in actions
    assert actions["RESAMPLE"].priority == "CRITICAL"
    assert "22050" in actions["RESAMPLE"].reason or "16000" in actions["RESAMPLE"].reason

    # 2. MONO_DOWNMIX must be triggered (stereo file exists)
    assert "MONO_DOWNMIX" in actions
    assert actions["MONO_DOWNMIX"].priority == "RECOMMENDED"

    # 3. PRUNE_SILENCE must be triggered
    assert "PRUNE_SILENCE" in actions
    assert actions["PRUNE_SILENCE"].priority == "CRITICAL"
    assert actions["PRUNE_SILENCE"].affected_files_count == 1

    # 4. PRUNE_OR_DECLIP must be triggered
    assert "PRUNE_OR_DECLIP" in actions
    assert actions["PRUNE_OR_DECLIP"].priority == "RECOMMENDED"
    assert actions["PRUNE_OR_DECLIP"].affected_files_count == 1

    # 5. CHUNKING_OR_PADDING must be triggered (1.0s to 6.0s disparity)
    assert "CHUNKING_OR_PADDING" in actions


def test_registry_dataset_profiler_dispatch(synthetic_dataset_dir):
    """Verify tool execution through the universal Layer 3 registry."""
    tool_names = get_registered_tool_names()
    assert "dataset_profiler" in tool_names

    res = execute_tool("dataset_profiler", args={"dataset_path": str(synthetic_dataset_dir)})
    assert res.status == "success"
    assert res.tool == "dataset_profiler"
    assert res.result["total_files"] == 6
    assert len(res.result["recommendations"]) >= 4
    assert "CRITICAL" in res.result["executive_summary"]


def test_empty_dataset_handling(tmp_path):
    """Verify clean abstention on empty directories."""
    empty_dir = tmp_path / "empty_folder"
    empty_dir.mkdir()

    res = execute_tool("dataset_profiler", args={"dataset_path": str(empty_dir)})
    assert res.status == "insufficient_samples"
    assert any("No valid audio files found" in w for w in res.warnings)

