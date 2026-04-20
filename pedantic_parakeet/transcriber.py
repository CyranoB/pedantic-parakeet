"""Transcription engine with backend selection.

This module provides:
- Token, Segment, TranscriptionResult dataclasses for transcription output
- Transcriber facade that delegates to backend implementations
- create_backend factory for instantiating backends
"""

from collections.abc import Callable
from pathlib import Path

from .backends.base import Backend, BaseTranscriber, STTCapabilities
from .backends.registry import resolve_model
from .types import Segment, Token, TranscriptionResult

# Re-export types for backwards compatibility
__all__ = [
    "Token",
    "Segment",
    "TranscriptionResult",
    "Transcriber",
    "create_backend",
    "DEFAULT_MODEL",
]

# Default model - optimized for Apple Silicon
DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def create_backend(
    model_id: str = DEFAULT_MODEL,
    backend: Backend | None = None,
    max_words_per_segment: int | None = None,
    max_segment_duration: float | None = None,
    chunk_duration: float = 120.0,
    overlap_duration: float = 15.0,
    language: str | None = None,
    language_strength: float = 0.5,
) -> BaseTranscriber:
    """Create a transcription backend instance.

    Args:
        model_id: HuggingFace model ID or alias (e.g., "parakeet", "whisper").
        backend: Override backend selection (auto-detected from registry by default).
        max_words_per_segment: Limit words per segment (for subtitle-friendly output).
        max_segment_duration: Limit segment duration in seconds.
        chunk_duration: Split long audio into chunks of this length (seconds).
        overlap_duration: Overlap between chunks to prevent word-cutting.
        language: Target language code (e.g., "fr") for language hints/bias.
        language_strength: Bias strength 0.0-2.0 for Parakeet backend.

    Returns:
        A backend implementing BaseTranscriber.

    Raises:
        ValueError: If model_id is not found in registry.
        RuntimeError: If mlx-audio backend requested but not installed.
    """
    # Resolve model from registry to get backend
    model_info = resolve_model(model_id)
    resolved_backend = backend if backend is not None else model_info.backend

    if resolved_backend == Backend.PARAKEET:
        from .backends.parakeet import ParakeetBackend

        return ParakeetBackend(
            model_id=model_info.model_id,
            max_words_per_segment=max_words_per_segment,
            max_segment_duration=max_segment_duration,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            language=language,
            language_strength=language_strength,
        )
    elif resolved_backend == Backend.MLX_AUDIO:
        from .backends.mlx_audio import MlxAudioBackend

        return MlxAudioBackend(
            model_id=model_info.model_id,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            language=language,
        )
    else:
        raise ValueError(f"Unknown backend: {resolved_backend}")


class Transcriber:
    """
    Audio transcriber facade with backend selection.

    Loads the model once and reuses it for multiple transcriptions.
    Delegates to the appropriate backend based on model selection.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        backend: Backend | None = None,
        max_words_per_segment: int | None = None,
        max_segment_duration: float | None = None,
        chunk_duration: float = 120.0,
        overlap_duration: float = 15.0,
        language: str | None = None,
        language_strength: float = 0.5,
    ):
        """
        Initialize the transcriber.

        Args:
            model_id: HuggingFace model ID or alias (e.g., "parakeet", "whisper").
            backend: Override backend selection (auto-detected from registry).
            max_words_per_segment: Limit words per segment (for subtitle-friendly output).
            max_segment_duration: Limit segment duration in seconds.
            chunk_duration: Split long audio into chunks of this length (seconds).
                            Use 0 to disable chunking (may cause memory issues).
            overlap_duration: Overlap between chunks to prevent word-cutting (seconds).
            language: Target language code to reduce code-switching (e.g., "fr").
            language_strength: Bias strength 0.0-2.0 (default 0.5).
        """
        # Resolve model to get actual model_id and backend
        try:
            model_info = resolve_model(model_id)
            self._resolved_model_id = model_info.model_id
            self._resolved_backend = backend if backend is not None else model_info.backend
        except ValueError:
            # Unknown model - assume Parakeet for backwards compatibility
            self._resolved_model_id = model_id
            self._resolved_backend = backend if backend is not None else Backend.PARAKEET

        # Store parameters for lazy backend creation
        self._backend_params = {
            "model_id": self._resolved_model_id,
            "backend": self._resolved_backend,
            "max_words_per_segment": max_words_per_segment,
            "max_segment_duration": max_segment_duration,
            "chunk_duration": chunk_duration,
            "overlap_duration": overlap_duration,
            "language": language,
            "language_strength": language_strength,
        }
        self._backend: BaseTranscriber | None = None

    @property
    def model_id(self) -> str:
        """The model identifier being used."""
        return self._resolved_model_id

    @property
    def capabilities(self) -> STTCapabilities:
        """The capabilities of the selected backend."""
        return self._get_backend().capabilities

    def _get_backend(self) -> BaseTranscriber:
        """Lazy create the backend on first use."""
        if self._backend is None:
            self._backend = create_backend(**self._backend_params)
        return self._backend

    def transcribe(
        self,
        audio_path: Path | str,
        chunk_callback: Callable[[float, float], None] | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file.
            chunk_callback: Optional callback(current_pos, total_pos) for progress.

        Returns:
            TranscriptionResult with text and timed segments.
        """
        return self._get_backend().transcribe(audio_path, chunk_callback)
