"""Sample Audio Generator for AuralQ Cockpit Demo Showcase."""

from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf


def ensure_showcase_samples(samples_dir: Optional[str | Path] = None) -> Path:
    """
    Ensure built-in demo audio samples exist on disk in data/samples/.
    Returns the resolved Path to the samples directory.
    """
    base_dir = Path(samples_dir or "data/samples")
    base_dir.mkdir(parents=True, exist_ok=True)

    sr = 22050

    # 1. Clean Sine 1000 Hz
    p_sine = base_dir / "clean_sine_1khz.wav"
    if not p_sine.exists():
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
        sf.write(str(p_sine), y, sr)

    # 2. Noisy Audio (Gaussian White Noise + 500 Hz tone)
    p_noise = base_dir / "noisy_audio.wav"
    if not p_noise.exists():
        t = np.linspace(0, 2.5, int(sr * 2.5), endpoint=False)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.25, int(sr * 2.5)).astype(np.float32)
        tone = (0.25 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)
        y = np.clip(noise + tone, -0.9, 0.9)
        sf.write(str(p_noise), y, sr)

    # 3. Clipped Audio (Saturation runs > 0.1%)
    p_clipped = base_dir / "clipped_snare.wav"
    if not p_clipped.exists():
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        tone = 1.6 * np.sin(2 * np.pi * 150.0 * t) * np.exp(-t * 2.0)
        y = np.clip(tone, -0.999, 0.999).astype(np.float32)
        sf.write(str(p_clipped), y, sr)

    # 4. Multi-Audio Sample Dataset Directory
    dataset_dir = base_dir / "sample_dataset"
    dataset_dir.mkdir(exist_ok=True)

    p_d1 = dataset_dir / "sample_44k_clean.wav"
    if not p_d1.exists():
        t = np.linspace(0, 2.0, 44100 * 2, endpoint=False)
        y = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(str(p_d1), y, 44100)

    p_d2 = dataset_dir / "sample_16k_speech.wav"
    if not p_d2.exists():
        t = np.linspace(0, 3.0, 16000 * 3, endpoint=False)
        y = (0.35 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        sf.write(str(p_d2), y, 16000)

    p_d3 = dataset_dir / "sample_stereo.wav"
    if not p_d3.exists():
        t = np.linspace(0, 1.5, sr * int(1.5), endpoint=False)
        left = 0.4 * np.sin(2 * np.pi * 600.0 * t)
        right = 0.4 * np.cos(2 * np.pi * 600.0 * t)
        stereo = np.column_stack([left, right]).astype(np.float32)
        sf.write(str(p_d3), stereo, sr)

    p_d4 = dataset_dir / "sample_silent.wav"
    if not p_d4.exists():
        y = np.zeros(sr * 1, dtype=np.float32)
        sf.write(str(p_d4), y, sr)

    return base_dir


if __name__ == "__main__":
    from typing import Optional
    path = ensure_showcase_samples()
    print(f"Showcase samples generated successfully in {path.resolve()}")
