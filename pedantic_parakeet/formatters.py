"""Output formatters for transcription results."""

import json
from datetime import datetime, timezone

from .transcriber import TranscriptionResult

# Schema version for JSON output (for future compatibility)
JSON_SCHEMA_VERSION = "1.0"


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm (comma for milliseconds)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm (dot for milliseconds)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_timestamp_simple(seconds: float) -> str:
    """Format seconds as simple timestamp: MM:SS or HH:MM:SS for longer audio."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_txt(
    result: TranscriptionResult, timestamps: bool = False, pause_threshold: float = 2.0
) -> str:
    """
    Format transcription as plain text.

    Args:
        result: Transcription result
        timestamps: If True, include timestamps for each segment
        pause_threshold: Insert paragraph break when gap between segments exceeds this (seconds)

    Returns:
        Plain text transcript with one segment per line and paragraph breaks on pauses
    """
    if not result.segments:
        return result.text

    lines = []
    prev_end = 0.0

    for segment in result.segments:
        # Add paragraph break if there's a long pause
        if lines and (segment.start - prev_end) > pause_threshold:
            lines.append("")

        if timestamps:
            ts = _format_timestamp_simple(segment.start)
            lines.append(f"[{ts}] {segment.text}")
        else:
            lines.append(segment.text)

        prev_end = segment.end

    return "\n".join(lines)


def format_srt(result: TranscriptionResult) -> str:
    """
    Format transcription as SRT (SubRip) subtitle format.

    Returns:
        SRT formatted string
    """
    lines = []
    for i, segment in enumerate(result.segments, start=1):
        start_ts = _format_timestamp_srt(segment.start)
        end_ts = _format_timestamp_srt(segment.end)
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(segment.text)
        lines.append("")  # Blank line between cues
    return "\n".join(lines)


def format_vtt(result: TranscriptionResult) -> str:
    """
    Format transcription as WebVTT subtitle format.

    Returns:
        VTT formatted string
    """
    lines = ["WEBVTT", ""]  # Header and blank line
    for segment in result.segments:
        start_ts = _format_timestamp_vtt(segment.start)
        end_ts = _format_timestamp_vtt(segment.end)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(segment.text)
        lines.append("")  # Blank line between cues
    return "\n".join(lines)


def format_json(result: TranscriptionResult) -> str:
    """
    Format transcription as structured JSON.

    Returns:
        JSON formatted string with full metadata
    """
    data = {
        "schema_version": JSON_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "audio_path": result.audio_path,
        "model_id": result.model_id,
        "text": result.text,
        "duration_seconds": result.duration,
        "segments": [
            {
                "text": seg.text,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "duration": round(seg.duration, 3),
                "confidence": round(seg.confidence, 3),
                "tokens": [
                    {
                        "text": tok.text,
                        "start": round(tok.start, 3),
                        "end": round(tok.end, 3),
                        "confidence": round(tok.confidence, 3),
                    }
                    for tok in seg.tokens
                ],
            }
            for seg in result.segments
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# Mapping of format names to formatter functions
FORMATTERS = {
    "txt": format_txt,
    "srt": format_srt,
    "vtt": format_vtt,
    "json": format_json,
}

# File extensions for each format
EXTENSIONS = {
    "txt": ".txt",
    "srt": ".srt",
    "vtt": ".vtt",
    "json": ".json",
}
