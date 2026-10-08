"""Layer 3: Deterministic DSP tools package for AuralQ."""

from src.tools.energy import rms_energy
from src.tools.registry import (
    TOOL_CATALOGUE,
    execute_tool,
    get_registered_tool_names,
    get_tool_metadata,
    list_tools_catalogue,
)
from src.tools.schemas import (
    BaseToolArgs,
    MFCCArgs,
    RMSEnergyArgs,
    SpectralBandwidthArgs,
    SpectralCentroidArgs,
    SpectralFlatnessArgs,
    SpectralRolloffArgs,
    SpectrogramArgs,
    ToolResultModel,
    ZeroCrossingRateArgs,
    generate_numeric_citation_aliases,
)
from src.tools.spectral import (
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_rolloff,
)
from src.tools.temporal import zero_crossing_rate
from src.tools.timbre import mfcc
from src.tools.visual import spectrogram

__all__ = [
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "mfcc",
    "spectrogram",
    "execute_tool",
    "get_registered_tool_names",
    "get_tool_metadata",
    "list_tools_catalogue",
    "TOOL_CATALOGUE",
    "ToolResultModel",
    "BaseToolArgs",
    "RMSEnergyArgs",
    "ZeroCrossingRateArgs",
    "SpectralCentroidArgs",
    "SpectralBandwidthArgs",
    "SpectralRolloffArgs",
    "SpectralFlatnessArgs",
    "MFCCArgs",
    "SpectrogramArgs",
    "generate_numeric_citation_aliases",
]

