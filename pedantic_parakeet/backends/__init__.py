"""Backend contracts and registry for STT transcription.

This module provides:
- Backend enum for selecting transcription engines
- STTCapabilities dataclass for capability flags
- ModelInfo dataclass for model metadata
- BaseTranscriber protocol for backend implementations
"""

from .base import (
    Backend,
    BaseTranscriber,
    ModelInfo,
    STTCapabilities,
)

__all__ = [
    "Backend",
    "BaseTranscriber",
    "ModelInfo",
    "STTCapabilities",
]
