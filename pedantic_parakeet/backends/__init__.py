"""Backend contracts and registry for STT transcription.

This module provides:
- Backend enum for selecting transcription engines
- STTCapabilities dataclass for capability flags
- ModelInfo dataclass for model metadata
- BaseTranscriber protocol for backend implementations
- ParakeetBackend implementation for Parakeet TDT models
- MlxAudioBackend implementation for mlx-audio models (optional)
"""

from .base import (
    Backend,
    BaseTranscriber,
    ModelInfo,
    STTCapabilities,
)
from .parakeet import ParakeetBackend

__all__ = [
    "Backend",
    "BaseTranscriber",
    "ModelInfo",
    "MlxAudioBackend",
    "ParakeetBackend",
    "STTCapabilities",
]


def __getattr__(name: str):
    """Lazy import MlxAudioBackend to avoid importing mlx-audio when not needed."""
    if name == "MlxAudioBackend":
        from .mlx_audio import MlxAudioBackend

        return MlxAudioBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
