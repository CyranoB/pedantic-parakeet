"""Transcription result types.

Contains dataclasses for transcription output that are shared between
the transcriber facade and backend implementations.
"""

from dataclasses import dataclass, field


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
