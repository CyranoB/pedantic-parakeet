"""mlx-audio backend implementation.

This backend uses the mlx-audio library for speech-to-text transcription,
supporting Whisper and other models available through mlx-audio.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..types import Segment, Token, TranscriptionResult
from .base import Backend, STTCapabilities
from .registry import MODEL_REGISTRY

if TYPE_CHECKING:
    pass

# Flag to track if mlx-audio is available
_mlx_audio_available: bool | None = None


def _check_mlx_audio_available() -> bool:
    """Check if mlx-audio is installed and available."""
    global _mlx_audio_available
    if _mlx_audio_available is None:
        try:
            import mlx_audio.stt  # noqa: F401

            _mlx_audio_available = True
        except ImportError:
            _mlx_audio_available = False
    return _mlx_audio_available


def is_mlx_audio_available() -> bool:
    """Check if mlx-audio is installed and available.

    Returns:
        True if mlx-audio can be imported, False otherwise.
    """
    return _check_mlx_audio_available()


class MlxAudioBackend:
    """Backend implementation for mlx-audio models.

    Uses the mlx-audio library for transcription. Supports Whisper models
    and other STT models available through mlx-audio.
    """

    def __init__(
        self,
        model_id: str,
        chunk_duration: float = 120.0,
        overlap_duration: float = 15.0,
        language: str | None = None,
    ):
        """Initialize the mlx-audio backend.

        Args:
            model_id: HuggingFace model ID (e.g., 'mlx-community/whisper-large-v3-turbo-asr-fp16').
            chunk_duration: Split long audio into chunks (seconds). 0 to disable.
            overlap_duration: Overlap between chunks to prevent word-cutting.
            language: Language hint for transcription (e.g., "en", "fr").

        Raises:
            RuntimeError: If mlx-audio is not installed.
        """
        if not _check_mlx_audio_available():
            raise RuntimeError(
                "mlx-audio is required for this backend but not installed. "
                "Install with: pip install 'pedantic-parakeet[mlx-audio]'"
            )

        self._model_id = model_id
        self.chunk_duration = chunk_duration if chunk_duration > 0 else None
        self.overlap_duration = overlap_duration
        self.language = language
        self._model: Any = None

        # Get capabilities from registry or use defaults
        model_info = MODEL_REGISTRY.get(model_id)
        if model_info is not None:
            self._capabilities = model_info.capabilities
        else:
            # Unknown model - assume reasonable defaults
            self._capabilities = STTCapabilities(
                supports_timestamps=True,
                supports_language_bias=False,
                supports_language_hint=True,
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
        return Backend.MLX_AUDIO

    def _load_model(self) -> Any:
        """Lazy load the model on first use."""
        if self._model is None:
            from mlx_audio.stt.utils import load_model

            self._model = load_model(self._model_id)
        return self._model

    def _build_generate_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for model.generate() based on capabilities."""
        kwargs: dict[str, Any] = {}

        # Language hint if supported
        if self._capabilities.supports_language_hint and self.language:
            kwargs["language"] = self.language

        # Word timestamps for Whisper models
        if self._capabilities.supports_timestamps:
            # Check if this is a Whisper model (they support word_timestamps)
            if "whisper" in self._model_id.lower():
                kwargs["word_timestamps"] = True

        # Chunking parameters if supported
        if self._capabilities.supports_chunking and self.chunk_duration:
            kwargs["chunk_duration"] = self.chunk_duration
            kwargs["overlap_duration"] = self.overlap_duration

        return kwargs

    def _convert_result(self, result: Any, audio_path: str) -> TranscriptionResult:
        """Convert mlx-audio output to TranscriptionResult.

        Handles various output formats from different mlx-audio models:
        - Whisper: result.segments with optional words
        - Parakeet v2: result.sentences with tokens
        - Text-only: result.text without timestamps
        """
        segments: list[Segment] = []

        # Try different result formats
        if hasattr(result, "segments") and result.segments:
            # Whisper-style output with segments
            for seg in result.segments:
                tokens: list[Token] = []

                # Extract word-level tokens if available
                words = getattr(seg, "words", None) or seg.get("words", []) if hasattr(seg, "get") else []
                for word in words:
                    if hasattr(word, "word"):
                        # Object-style word
                        tokens.append(
                            Token(
                                text=word.word,
                                start=getattr(word, "start", 0.0),
                                end=getattr(word, "end", 0.0),
                                confidence=getattr(word, "probability", 1.0),
                            )
                        )
                    elif isinstance(word, dict):
                        # Dict-style word
                        tokens.append(
                            Token(
                                text=word.get("word", word.get("text", "")),
                                start=word.get("start", 0.0),
                                end=word.get("end", 0.0),
                                confidence=word.get("probability", word.get("confidence", 1.0)),
                            )
                        )

                # Extract segment info
                if hasattr(seg, "text"):
                    seg_text = seg.text
                    seg_start = getattr(seg, "start", 0.0)
                    seg_end = getattr(seg, "end", 0.0)
                elif isinstance(seg, dict):
                    seg_text = seg.get("text", "")
                    seg_start = seg.get("start", 0.0)
                    seg_end = seg.get("end", 0.0)
                else:
                    continue

                segments.append(
                    Segment(
                        text=seg_text.strip() if seg_text else "",
                        start=seg_start,
                        end=seg_end,
                        confidence=1.0,
                        tokens=tokens,
                    )
                )

        elif hasattr(result, "sentences") and result.sentences:
            # Parakeet-style output with sentences
            for sent in result.sentences:
                tokens: list[Token] = []
                sent_tokens = getattr(sent, "tokens", [])
                for tok in sent_tokens:
                    tokens.append(
                        Token(
                            text=getattr(tok, "text", ""),
                            start=getattr(tok, "start", 0.0),
                            end=getattr(tok, "end", 0.0),
                            confidence=getattr(tok, "confidence", 1.0),
                        )
                    )

                segments.append(
                    Segment(
                        text=getattr(sent, "text", "").strip(),
                        start=getattr(sent, "start", 0.0),
                        end=getattr(sent, "end", 0.0),
                        confidence=getattr(sent, "confidence", 1.0),
                        tokens=tokens,
                    )
                )

        # Extract full text
        if hasattr(result, "text"):
            full_text = result.text
        elif segments:
            full_text = " ".join(seg.text for seg in segments)
        else:
            full_text = ""

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            audio_path=audio_path,
            model_id=self._model_id,
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
                Note: Not all mlx-audio models support progress callbacks.

        Returns:
            TranscriptionResult with text and timed segments.
        """
        model = self._load_model()
        kwargs = self._build_generate_kwargs()

        # Run transcription
        result = model.generate(str(audio_path), **kwargs)

        return self._convert_result(result, str(audio_path))
