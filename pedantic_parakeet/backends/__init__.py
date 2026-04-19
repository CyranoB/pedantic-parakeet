"""Backend contracts and registry for STT transcription.

This module provides:
- Backend enum for selecting transcription engines
- STTCapabilities dataclass for capability flags
- ModelInfo dataclass for model metadata
- BaseTranscriber protocol for backend implementations
- ParakeetBackend implementation for Parakeet TDT models
- MlxAudioBackend implementation for mlx-audio models (optional)
"""

from .base import Backend, BaseTranscriber, ModelInfo, STTCapabilities

__all__ = [
    "Backend",
    "BaseTranscriber",
    "ModelInfo",
    "MlxAudioBackend",
    "ParakeetBackend",
    "STTCapabilities",
]


def __getattr__(name: str):
    """Lazy import backend implementations to avoid eager MLX initialization."""
    if name == "ParakeetBackend":
        from .parakeet import ParakeetBackend

        return ParakeetBackend
    if name == "MlxAudioBackend":
        from .mlx_audio import MlxAudioBackend

        return MlxAudioBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
