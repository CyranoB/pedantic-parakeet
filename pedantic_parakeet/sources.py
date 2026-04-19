"""Input resolution and URL media acquisition helpers."""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .audio import discover_audio_files, is_supported_audio


class InputResolutionError(ValueError):
    """Raised when CLI inputs cannot be resolved into media sources."""


@dataclass(frozen=True)
class ResolvedSource:
    """A resolved CLI input before any temporary download occurs."""

    source_kind: str
    original_input: str
    display_name: str
    output_stem: str
    media_path: Path | None


@dataclass(frozen=True)
class MaterializedSource:
    """A local media path ready for transcription."""

    display_name: str
    output_stem: str
    media_path: Path
    is_temporary: bool
    cleanup_path: Path | None


def is_url_input(value: str) -> bool:
    """Return True when the input looks like an HTTP(S) URL."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def sanitize_output_stem(value: str) -> str:
    """Turn a title or slug into a safe output filename stem."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:120] or "transcript"


def fetch_url_metadata(url: str) -> dict:
    """Fetch media metadata for a public URL without downloading the file."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise InputResolutionError(
            "URL inputs require yt-dlp. Install pedantic-parakeet with yt-dlp available."
        ) from exc

    options = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
    }
    with YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=False)


def _title_from_metadata(url: str, info: dict) -> str:
    title = info.get("title") or info.get("id")
    if title:
        return sanitize_output_stem(str(title))

    parsed = urlparse(url)
    slug = Path(parsed.path).name or parsed.netloc
    return sanitize_output_stem(slug)


def resolve_inputs(inputs: list[str], recursive: bool = False) -> list[ResolvedSource]:
    """Resolve raw CLI inputs into local or URL sources."""
    sources: list[ResolvedSource] = []

    for raw_input in inputs:
        if is_url_input(raw_input):
            info = fetch_url_metadata(raw_input)
            output_stem = _title_from_metadata(raw_input, info)
            sources.append(
                ResolvedSource(
                    source_kind="url",
                    original_input=raw_input,
                    display_name=info.get("title") or output_stem,
                    output_stem=output_stem,
                    media_path=None,
                )
            )
            continue

        path = Path(raw_input).expanduser()
        if not path.exists():
            raise InputResolutionError(f"Input does not exist: {raw_input}")

        if path.is_file():
            if is_supported_audio(path):
                sources.append(
                    ResolvedSource(
                        source_kind="local",
                        original_input=raw_input,
                        display_name=path.name,
                        output_stem=path.stem,
                        media_path=path,
                    )
                )
            continue

        for media_path in discover_audio_files([path], recursive=recursive):
            sources.append(
                ResolvedSource(
                    source_kind="local",
                    original_input=raw_input,
                    display_name=media_path.name,
                    output_stem=media_path.stem,
                    media_path=media_path,
                )
            )

    return sources


def _pick_downloaded_media(temp_dir: Path) -> Path:
    candidates = [path for path in temp_dir.rglob("*") if path.is_file()]
    media_candidates = [path for path in candidates if is_supported_audio(path)]
    if media_candidates:
        return min(media_candidates)
    if candidates:
        return min(candidates)
    raise InputResolutionError("yt-dlp did not produce a downloadable media file")


def download_url_source(source: ResolvedSource) -> MaterializedSource:
    """Download a public URL to a temporary local media file."""
    if source.source_kind != "url":
        raise InputResolutionError("Only URL sources can be downloaded")

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise InputResolutionError(
            "URL inputs require yt-dlp. Install pedantic-parakeet with yt-dlp available."
        ) from exc

    temp_dir = Path(tempfile.mkdtemp(prefix="pedantic-parakeet-"))
    options = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "format": "bestaudio/best/best",
        "outtmpl": str(temp_dir / "%(title).120B [%(id)s].%(ext)s"),
    }

    try:
        with YoutubeDL(options) as ydl:
            ydl.extract_info(source.original_input, download=True)
        media_path = _pick_downloaded_media(temp_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return MaterializedSource(
        display_name=source.display_name,
        output_stem=source.output_stem,
        media_path=media_path,
        is_temporary=True,
        cleanup_path=temp_dir,
    )


def materialize_source(source: ResolvedSource) -> MaterializedSource:
    """Return a local media path for a resolved source."""
    if source.source_kind == "url":
        return download_url_source(source)

    if source.media_path is None:
        raise InputResolutionError("Local source is missing a media path")

    return MaterializedSource(
        display_name=source.display_name,
        output_stem=source.output_stem,
        media_path=source.media_path,
        is_temporary=False,
        cleanup_path=None,
    )


def cleanup_materialized_source(source: MaterializedSource) -> None:
    """Remove temporary downloads created for URL transcription."""
    if source.is_temporary and source.cleanup_path is not None:
        shutil.rmtree(source.cleanup_path, ignore_errors=True)
