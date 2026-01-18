"""Transcription engine wrapping parakeet-mlx."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .language_bias import build_language_bias
from .parakeet_mlx import (
    BaseParakeet,
    DecodingConfig,
    Greedy,
    SentenceConfig,
    from_pretrained,
)

# Default model - optimized for Apple Silicon
DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


@dataclass
class Token:
    """A word or subword token with timing information."""

    text: str
    start: float  # seconds
    end: float  # seconds
    confidence: float = 1.0


@dataclass
class Segment:
    """A sentence or phrase segment with timing information."""

    text: str
    start: float  # seconds
    end: float  # seconds
    confidence: float = 1.0
    tokens: list[Token] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    text: str
    segments: list[Segment]
    audio_path: str
    model_id: str

    @property
    def duration(self) -> float:
        """Total duration based on last segment end time."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end


class Transcriber:
    """
    Audio transcriber using Parakeet TDT models.

    Loads the model once and reuses it for multiple transcriptions.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
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
            model_id: HuggingFace model ID
            max_words_per_segment: Limit words per segment (for subtitle-friendly output)
            max_segment_duration: Limit segment duration in seconds
            chunk_duration: Split long audio into chunks of this length (seconds).
                            Use 0 to disable chunking (may cause memory issues).
            overlap_duration: Overlap between chunks to prevent word-cutting (seconds)
            language: Target language code to reduce code-switching (e.g., "fr")
            language_strength: Bias strength 0.0-2.0 (default 0.5)
        """
        self.model_id = model_id
        self.max_words = max_words_per_segment
        self.max_duration = max_segment_duration
        self.chunk_duration = chunk_duration if chunk_duration > 0 else None
        self.overlap_duration = overlap_duration
        self.language = language
        self.language_strength = language_strength
        self._model: BaseParakeet | None = None

    def _load_model(self) -> BaseParakeet:
        """Lazy load the model on first use."""
        if self._model is None:
            self._model = from_pretrained(self.model_id)
        return self._model

    def _build_config(self, model: BaseParakeet) -> DecodingConfig:
        """Build decoding configuration."""
        language_bias = None
        if self.language:
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
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            chunk_callback: Optional callback(current_pos, total_pos) for progress

        Returns:
            TranscriptionResult with text and timed segments
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
            model_id=self.model_id,
        )
