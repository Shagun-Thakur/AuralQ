"""Sample Audio Generator for AuralQ Cockpit Demo Showcase."""

from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf


SHOWCASE_VERSION = "v3_smooth_hann_fade"


def _apply_linear_fade(y: np.ndarray, sr: int, fade_ms: float = 50.0) -> np.ndarray:
    """
    Apply a 50 ms smooth raised-cosine (half-Hann) fade-in and fade-out to built-in showcase signals
    to eliminate boundary click artifacts and spectral leakage across the STFT window.
    """
    fade_len = int(sr * (fade_ms / 1000.0))
    if len(y) <= 2 * fade_len or fade_len <= 0:
        return y
    y_faded = y.copy()
    # Raised cosine (Hann) taper: continuous 1st derivative eliminates spectral splatter
    phase = np.linspace(0.0, np.pi, fade_len, endpoint=False, dtype=np.float32)
    fade_in = 0.5 * (1.0 - np.cos(phase))
    fade_out = 0.5 * (1.0 + np.cos(phase))
    y_faded[:fade_len] *= fade_in
    y_faded[-fade_len:] *= fade_out
    return y_faded


def ensure_showcase_samples(samples_dir: Optional[str | Path] = None) -> Path:
    """
    Ensure built-in demo audio samples exist on disk in data/samples/.
    Regenerates when the version marker changes or when sample files are missing.
    Returns the resolved Path to the samples directory.
    """
    base_dir = Path(samples_dir or "data/samples")
    base_dir.mkdir(parents=True, exist_ok=True)

    version_file = base_dir / ".showcase_version"
    needs_generation = False

    if not version_file.exists():
        needs_generation = True
    else:
        try:
            if version_file.read_text(encoding="utf-8").strip() != SHOWCASE_VERSION:
                needs_generation = True
        except Exception:
            needs_generation = True

    required_files = [
        base_dir / "clean_sine_1khz.wav",
        base_dir / "noisy_audio.wav",
        base_dir / "clipped_snare.wav",
    ]
    if any(not f.exists() for f in required_files):
        needs_generation = True

    if not needs_generation:
        return base_dir

    sr = 22050

    # 1. Clean Sine 1000 Hz (with 10 ms linear fade-in and fade-out to prevent click artifacts)
    p_sine = base_dir / "clean_sine_1khz.wav"
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y_sine = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    y_sine = _apply_linear_fade(y_sine, sr, fade_ms=10.0)
    sf.write(str(p_sine), y_sine, sr)

    # 2. Noisy Audio (Gaussian White Noise + 500 Hz tone with 10 ms linear fade)
    p_noise = base_dir / "noisy_audio.wav"
    t = np.linspace(0, 2.5, int(sr * 2.5), endpoint=False)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.25, int(sr * 2.5)).astype(np.float32)
    tone = (0.25 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)
    y_noise = np.clip(noise + tone, -0.9, 0.9)
    y_noise = _apply_linear_fade(y_noise, sr, fade_ms=10.0)
    sf.write(str(p_noise), y_noise, sr)

    # 3. Clipped Audio (Saturation runs > 0.1% with 10 ms linear fade)
    p_clipped = base_dir / "clipped_snare.wav"
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    tone = 1.6 * np.sin(2 * np.pi * 150.0 * t) * np.exp(-t * 2.0)
    y_clipped = np.clip(tone, -0.999, 0.999).astype(np.float32)
    y_clipped = _apply_linear_fade(y_clipped, sr, fade_ms=10.0)
    sf.write(str(p_clipped), y_clipped, sr)

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

    # Write version marker file
    try:
        version_file.write_text(SHOWCASE_VERSION, encoding="utf-8")
    except Exception:
        pass

    return base_dir


if __name__ == "__main__":
    from typing import Optional
    path = ensure_showcase_samples()
    print(f"Showcase samples generated successfully in {path.resolve()}")
