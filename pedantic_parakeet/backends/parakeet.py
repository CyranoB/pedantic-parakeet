"""Parakeet TDT backend implementation.

This backend wraps the vendored parakeet-mlx implementation for high-accuracy
speech-to-text transcription with language bias support.
"""

from collections.abc import Callable
from pathlib import Path

from ..language_bias import build_language_bias
from ..parakeet_mlx import (
    BaseParakeet,
    DecodingConfig,
    Greedy,
    SentenceConfig,
    from_pretrained,
)
from ..types import Segment, Token, TranscriptionResult
from .base import Backend, STTCapabilities
from .registry import MODEL_REGISTRY

# Default Parakeet model
DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


class ParakeetBackend:
    """Backend implementation for Parakeet TDT models.

    Uses the vendored parakeet-mlx implementation for transcription.
    Supports language biasing to reduce code-switching in multilingual audio.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_PARAKEET_MODEL,
        max_words_per_segment: int | None = None,
        max_segment_duration: float | None = None,
        chunk_duration: float = 120.0,
        overlap_duration: float = 15.0,
        language: str | None = None,
        language_strength: float = 0.5,
    ):
        """Initialize the Parakeet backend.

        Args:
            model_id: HuggingFace model ID for Parakeet model.
            max_words_per_segment: Limit words per segment (subtitle-friendly).
            max_segment_duration: Limit segment duration in seconds.
            chunk_duration: Split long audio into chunks of this length (seconds).
                            Use 0 to disable chunking (may cause memory issues).
            overlap_duration: Overlap between chunks to prevent word-cutting.
            language: Target language code to reduce code-switching (e.g., "fr").
            language_strength: Bias strength 0.0-2.0 (default 0.5).
        """
        self._model_id = model_id
        self.max_words = max_words_per_segment
        self.max_duration = max_segment_duration
        self.chunk_duration = chunk_duration if chunk_duration > 0 else None
        self.overlap_duration = overlap_duration
        self.language = language
        self.language_strength = language_strength
        self._model: BaseParakeet | None = None

        # Get capabilities from registry or use defaults
        model_info = MODEL_REGISTRY.get(model_id)
        if model_info is not None:
            self._capabilities = model_info.capabilities
        else:
            # Unknown Parakeet model - assume full capabilities
            self._capabilities = STTCapabilities(
                supports_timestamps=True,
                supports_language_bias=True,
                supports_language_hint=False,
                supports_chunking=True,
            )

    @property
    def model_id(self) -> str:
        """The model identifier being used."""
        return self._model_id

    @property
    def capabilities(self) -> STTCapabilities:
        """The capabilities of this transcriber."""
        return self._capabilities

    @property
    def backend(self) -> Backend:
        """The backend type."""
        return Backend.PARAKEET

    def _load_model(self) -> BaseParakeet:
        """Lazy load the model on first use."""
        if self._model is None:
            self._model = from_pretrained(self._model_id)
        return self._model

    def _build_config(self, model: BaseParakeet) -> DecodingConfig:
        """Build decoding configuration."""
        language_bias = None
        if self.language and self._capabilities.supports_language_bias:
            language_bias = build_language_bias(
                model.vocabulary,
                self.language,
                self.language_strength,
            )

        return DecodingConfig(
            decoding=Greedy(),
            sentence=SentenceConfig(
                max_words=self.max_words,
                max_duration=self.max_duration,
            ),
            language_bias=language_bias,
        )

    def transcribe(
        self,
        audio_path: Path | str,
        chunk_callback: Callable[[float, float], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file.
            chunk_callback: Optional callback(current_pos, total_pos) for progress.

        Returns:
            TranscriptionResult with text and timed segments.
        """
        model = self._load_model()
        config = self._build_config(model)

        # Run transcription with chunking for long audio
        result = model.transcribe(
            str(audio_path),
            decoding_config=config,
            chunk_duration=self.chunk_duration,
            overlap_duration=self.overlap_duration,
            chunk_callback=chunk_callback,
        )

        # Convert to our data structures
        segments = []
        for sent in result.sentences:
            tokens = [
                Token(
                    text=tok.text,
                    start=tok.start,
                    end=tok.end,
                    confidence=tok.confidence,
                )
                for tok in sent.tokens
            ]
            segments.append(
                Segment(
                    text=sent.text.strip(),
                    start=sent.start,
                    end=sent.end,
                    confidence=sent.confidence,
                    tokens=tokens,
                )
            )

        return TranscriptionResult(
            text=result.text,
            segments=segments,
            audio_path=str(audio_path),
            model_id=self._model_id,
        )
