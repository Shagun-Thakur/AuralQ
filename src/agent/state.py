"""Session state management, multi-turn history tracking, and deterministic tool result caching."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.agent.schemas import SessionSummary, TurnRecord
from src.audio.validation import AudioMetadata
from src.tools.schemas import ToolResultModel


def serialize_tool_cache_key(audio_id: str, tool_name: str, parameters: Dict[str, Any]) -> str:
    """
    Generate a deterministic, parameter-order-invariant cache key for a tool calculation.
    Example: 'clip1.wav:spectral_centroid:{"hop_length":512,"n_fft":2048}'
    """
    sorted_params = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return f"{audio_id}:{tool_name}:{sorted_params}"


class SessionState:
    """
    State manager for an active AuralQ conversational session.
    Tracks loaded audio, datasets, execution history, and caches deterministic tool results.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self.audio_metadata: Optional[AudioMetadata] = None
        self.waveform: Optional[np.ndarray] = None
        self.sr: Optional[int] = None
        self.dataset_path: Optional[str] = None
        self._tool_cache: Dict[str, ToolResultModel] = {}
        self.turns: List[TurnRecord] = []

    @property
    def audio_id(self) -> str:
        """Return the active audio identifier or filename."""
        if self.audio_metadata and self.audio_metadata.file_path:
            return self.audio_metadata.file_path
        if self.dataset_path:
            return self.dataset_path
        return "unloaded"

    def set_audio(
        self,
        metadata: AudioMetadata,
        waveform: np.ndarray,
        sr: int,
    ) -> None:
        """Set or update the active audio recording, invalidating previous audio tool caches."""
        self.audio_metadata = metadata
        self.waveform = waveform
        self.sr = sr
        # Invalidate tool cache when a new audio file is loaded
        self._tool_cache.clear()

    def set_dataset(self, path: str) -> None:
        """Set or update the active dataset path, clearing dataset tool cache."""
        self.dataset_path = path
        # Invalidate dataset profiler cache entries
        keys_to_delete = [k for k in self._tool_cache if "dataset_profiler" in k]
        for k in keys_to_delete:
            del self._tool_cache[k]

    def clear_active_inputs(self) -> None:
        """Reset loaded audio and dataset references and clear the tool cache."""
        self.audio_metadata = None
        self.waveform = None
        self.sr = None
        self.dataset_path = None
        self._tool_cache.clear()

    def get_cached_tool_result(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> Optional[ToolResultModel]:
        """Retrieve a cached tool execution result if available for current active audio."""
        key = serialize_tool_cache_key(self.audio_id, tool_name, parameters)
        return self._tool_cache.get(key)

    def cache_tool_result(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: ToolResultModel,
    ) -> None:
        """Cache a successful deterministic tool result."""
        # Only cache successful DSP results
        if result.status == "success":
            key = serialize_tool_cache_key(self.audio_id, tool_name, parameters)
            self._tool_cache[key] = result

    def add_turn(self, turn: TurnRecord) -> None:
        """Record a completed conversational turn."""
        self.turns.append(turn)

    def get_recent_history_context(self, max_turns: int = 3) -> str:
        """
        Format the last `max_turns` conversational turns into compact dialogue context.
        Designed to fit comfortably within the 2048-token context window.
        """
        if not self.turns:
            return ""

        recent = self.turns[-max_turns:]
        lines = ["CONVERSATION HISTORY:"]
        for t in recent:
            lines.append(f"User: {t.query}")
            lines.append(f"AuralQ: {t.response_text}")
        lines.append("")
        return "\n".join(lines)

    def get_summary(self) -> SessionSummary:
        """Produce a lightweight summary of the current session state."""
        return SessionSummary(
            session_id=self.session_id,
            turn_count=len(self.turns),
            active_audio_path=self.audio_metadata.file_path if self.audio_metadata else None,
            active_dataset_path=self.dataset_path,
            cached_tools=list(self._tool_cache.keys()),
        )

