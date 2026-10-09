"""Strict Tool Registry allowlist and execution dispatcher for AuralQ Layer 3."""

from typing import Any, Callable, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, ValidationError

from src.tools.dataset_advisor import dataset_profiler_tool
from src.tools.energy import rms_energy
from src.tools.schemas import (
    BaseToolArgs,
    DatasetProfileArgs,
    MFCCArgs,
    RMSEnergyArgs,
    SpectralBandwidthArgs,
    SpectralCentroidArgs,
    SpectralFlatnessArgs,
    SpectralRolloffArgs,
    SpectrogramArgs,
    ToolResultModel,
    ZeroCrossingRateArgs,
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


class ToolDefinition(BaseModel):
    """Metadata specification for an individual DSP tool."""
    name: str
    category: str
    description: str
    args_schema: Any
    intent_keywords: List[str]


TOOL_CATALOGUE: Dict[str, Dict[str, Any]] = {
    "rms_energy": {
        "func": rms_energy,
        "args_class": RMSEnergyArgs,
        "category": "energy",
        "description": "Compute root-mean-square energy and loudness stability statistics across time frames.",
        "intent_keywords": ["loudness", "volume", "consistent", "quiet", "amplitude", "dynamic range", "fluctuating"],
    },
    "zero_crossing_rate": {
        "func": zero_crossing_rate,
        "args_class": ZeroCrossingRateArgs,
        "category": "temporal",
        "description": "Compute frame-by-frame rate of sign changes, detecting percussiveness, noise, or smoothness.",
        "intent_keywords": ["smooth", "percussive", "noisy", "transient", "varying", "rapidity"],
    },
    "spectral_centroid": {
        "func": spectral_centroid,
        "args_class": SpectralCentroidArgs,
        "category": "spectral",
        "description": "Compute the weighted mean frequency of the spectrum, corresponding to perceived brightness.",
        "intent_keywords": ["bright", "dull", "sharp", "dark", "center frequency", "warm", "brightness"],
    },
    "spectral_bandwidth": {
        "func": spectral_bandwidth,
        "args_class": SpectralBandwidthArgs,
        "category": "spectral",
        "description": "Compute the spectral spread/width of frequencies around the centroid.",
        "intent_keywords": ["spread", "wide", "narrow", "frequency range", "bandwidth"],
    },
    "spectral_rolloff": {
        "func": spectral_rolloff,
        "args_class": SpectralRolloffArgs,
        "category": "spectral",
        "description": "Compute the frequency below which a specified percentage (e.g. 85%) of total spectral energy lies.",
        "intent_keywords": ["high frequency", "cutoff", "energy concentration", "rolloff"],
    },
    "spectral_flatness": {
        "func": spectral_flatness,
        "args_class": SpectralFlatnessArgs,
        "category": "spectral",
        "description": "Compute spectral flatness (ratio of geometric to arithmetic mean), measuring tonality vs noise.",
        "intent_keywords": ["tonal", "noise", "hiss", "harmonic", "pure tone", "noise-like"],
    },
    "mfcc": {
        "func": mfcc,
        "args_class": MFCCArgs,
        "category": "timbre",
        "description": "Extract Mel-Frequency Cepstral Coefficients (MFCCs) capturing timbral color and texture.",
        "intent_keywords": ["timbre", "texture", "color", "instrument", "spectral envelope"],
    },
    "spectrogram": {
        "func": spectrogram,
        "args_class": SpectrogramArgs,
        "category": "visual",
        "description": "Compute STFT time-frequency magnitude summaries and optional plot generation.",
        "intent_keywords": ["spectrogram", "visualize", "time-frequency", "plot", "display frequencies"],
    },
    "dataset_profiler": {
        "func": dataset_profiler_tool,
        "args_class": DatasetProfileArgs,
        "category": "dataset",
        "description": "Stream-audit an audio dataset folder, profiling acoustic distributions and generating deterministic preprocessing advice.",
        "intent_keywords": ["dataset", "folder", "batch", "profile", "preprocess", "preprocessing", "audit", "recommendation", "clean"],
    },
}


def get_registered_tool_names() -> List[str]:
    """Return all valid tool names in the allowlist."""
    return list(TOOL_CATALOGUE.keys())


def get_tool_metadata(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve metadata description and schema for a specific tool."""
    item = TOOL_CATALOGUE.get(name)
    if not item:
        return None
    return {
        "name": name,
        "category": item["category"],
        "description": item["description"],
        "parameters": item["args_class"].model_json_schema(),
        "intent_keywords": item["intent_keywords"],
    }


def list_tools_catalogue() -> List[Dict[str, Any]]:
    """List full tool specifications for all 8 registered tools."""
    return [get_tool_metadata(name) for name in TOOL_CATALOGUE]  # type: ignore


def execute_tool(
    name: str,
    waveform: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    args: Optional[Dict[str, Any]] = None,
) -> ToolResultModel:
    """
    Execute a registered DSP tool against a waveform or dataset with strict argument validation.
    Never raises unhandled exceptions.
    """
    if name not in TOOL_CATALOGUE:
        return ToolResultModel(
            tool=name,
            status="dsp_error",
            parameters=args or {},
            result={},
            perceptual_anchors={},
            citation_aliases={},
            execution_time_ms=0.0,
            warnings=[f"Tool '{name}' is not in the registered allowlist: {get_registered_tool_names()}"],
        )

    tool_entry = TOOL_CATALOGUE[name]
    args_class: Type[BaseToolArgs] = tool_entry["args_class"]
    func: Callable = tool_entry["func"]

    # 1. Validate arguments using Pydantic v2
    try:
        validated_args = args_class(**(args or {}))
    except ValidationError as ve:
        return ToolResultModel(
            tool=name,
            status="dsp_error",
            parameters=args or {},
            result={},
            perceptual_anchors={},
            citation_aliases={},
            execution_time_ms=0.0,
            warnings=[f"Invalid tool arguments for '{name}': {ve.errors()}"],
        )

    # 2. Execute deterministic function
    try:
        if name == "dataset_profiler":
            # Dataset profiler operates on filesystem path, not a preloaded waveform
            return func(
                dataset_path=validated_args.dataset_path,
                target_sr=validated_args.target_sr or 22050,
                max_files=validated_args.max_files,
            )

        if waveform is None or sr is None:
            return ToolResultModel(
                tool=name,
                status="dsp_error",
                parameters=validated_args.model_dump(),
                result={},
                perceptual_anchors={},
                citation_aliases={},
                execution_time_ms=0.0,
                warnings=[f"Tool '{name}' requires an ingested waveform and sample rate."],
            )

        return func(waveform=waveform, sr=sr, args=validated_args)
    except Exception as e:
        return ToolResultModel(
            tool=name,
            status="dsp_error",
            parameters=validated_args.model_dump(),
            result={},
            perceptual_anchors={},
            citation_aliases={},
            execution_time_ms=0.0,
            warnings=[f"Unexpected DSP execution failure in '{name}': {str(e)}"],
        )

