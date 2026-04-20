"""Backend capability contracts for STT transcription."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable
    from ..types import TranscriptionResult


class Backend(str, Enum):
    """Supported transcription backends."""

    PARAKEET = "parakeet"
    MLX_AUDIO = "mlx-audio"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class STTCapabilities:
    """Capability flags for speech-to-text backends.

    Describes what features a backend/model combination supports.
    """

    supports_timestamps: bool = True
    """Whether the model provides word/segment timestamps."""

    supports_language_bias: bool = False
    """Whether the model supports token-level language biasing (Parakeet-specific)."""

    supports_language_hint: bool = False
    """Whether the model accepts a language hint for transcription."""

    supports_chunking: bool = True
    """Whether the model supports chunked processing for long audio."""


@dataclass
class ModelInfo:
    """Metadata for a curated STT model.

    Contains all information needed to select and configure a model.
    """

    model_id: str
    """HuggingFace model identifier (e.g., 'mlx-community/parakeet-tdt-0.6b-v3')."""

    backend: Backend
    """Which transcription backend to use."""

    capabilities: STTCapabilities
    """What features this model supports."""

    description: str = ""
    """Human-readable description for CLI display."""

    aliases: list[str] = field(default_factory=list)
    """Short names for CLI convenience (e.g., ['parakeet', 'v3'])."""


@runtime_checkable
class BaseTranscriber(Protocol):
    """Protocol defining the transcriber interface.

    All backend implementations must satisfy this protocol.
    """

    @property
    def model_id(self) -> str:
        """The model identifier being used."""
        ...

    @property
    def capabilities(self) -> STTCapabilities:
        """The capabilities of this transcriber."""
        ...

    def transcribe(
        self,
        audio_path: "Path | str",
        chunk_callback: "Callable[[float, float], None] | None" = None,
    ) -> "TranscriptionResult":
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file to transcribe.
            chunk_callback: Optional progress callback(current_pos, total_duration).

        Returns:
            TranscriptionResult with text and timed segments.
        """
        ...
