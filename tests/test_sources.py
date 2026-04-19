"""Tests for media source resolution and URL acquisition."""

from pathlib import Path

import pytest

from pedantic_parakeet.sources import (
    InputResolutionError,
    MaterializedSource,
    ResolvedSource,
    cleanup_materialized_source,
    is_url_input,
    materialize_source,
    resolve_inputs,
)


def test_is_url_input_recognizes_http_and_https():
    assert is_url_input("https://example.com/video")
    assert is_url_input("http://example.com/video")
    assert not is_url_input("/tmp/video.mp4")
    assert not is_url_input("recording.m4a")


def test_resolve_inputs_expands_directories_and_local_files(tmp_path: Path):
    audio_file = tmp_path / "clip.m4a"
    audio_file.write_text("x", encoding="utf-8")
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_file = video_dir / "lesson.mp4"
    video_file.write_text("x", encoding="utf-8")

    sources = resolve_inputs([str(audio_file), str(video_dir)], recursive=False)

    assert [source.display_name for source in sources] == ["clip.m4a", "lesson.mp4"]
    assert [source.media_path for source in sources] == [audio_file, video_file]
    assert all(source.source_kind == "local" for source in sources)


def test_resolve_inputs_raises_for_missing_local_paths():
    with pytest.raises(InputResolutionError, match="Input does not exist"):
        resolve_inputs(["/tmp/does-not-exist-12345.mp4"])


def test_resolve_inputs_uses_metadata_for_public_urls(monkeypatch: pytest.MonkeyPatch):
    def fake_fetch(url: str) -> dict[str, str]:
        assert url == "https://example.com/watch?v=123"
        return {"title": "Bonjour le monde", "id": "123"}

    monkeypatch.setattr("pedantic_parakeet.sources.fetch_url_metadata", fake_fetch)

    sources = resolve_inputs(["https://example.com/watch?v=123"])

    assert sources == [
        ResolvedSource(
            source_kind="url",
            original_input="https://example.com/watch?v=123",
            display_name="Bonjour le monde",
            output_stem="Bonjour le monde",
            media_path=None,
        )
    ]


def test_resolve_inputs_allows_mixed_local_and_url_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    local_file = tmp_path / "sample.mov"
    local_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "pedantic_parakeet.sources.fetch_url_metadata",
        lambda url: {"title": "Conference talk", "id": "abc123"},
    )

    sources = resolve_inputs([str(local_file), "https://example.com/talk"])

    assert len(sources) == 2
    assert sources[0].media_path == local_file
    assert sources[1].source_kind == "url"
    assert sources[1].output_stem == "Conference talk"


def test_materialize_source_returns_local_source_unchanged(tmp_path: Path):
    media_path = tmp_path / "lesson.mp4"
    media_path.write_text("x", encoding="utf-8")
    source = ResolvedSource(
        source_kind="local",
        original_input=str(media_path),
        display_name=media_path.name,
        output_stem=media_path.stem,
        media_path=media_path,
    )

    materialized = materialize_source(source)

    assert materialized == MaterializedSource(
        display_name=media_path.name,
        output_stem=media_path.stem,
        media_path=media_path,
        is_temporary=False,
        cleanup_path=None,
    )


def test_materialize_source_downloads_url_to_temporary_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    downloaded_file = tmp_path / "downloaded.webm"
    downloaded_file.write_text("x", encoding="utf-8")

    def fake_download(source: ResolvedSource) -> MaterializedSource:
        return MaterializedSource(
            display_name=source.display_name,
            output_stem=source.output_stem,
            media_path=downloaded_file,
            is_temporary=True,
            cleanup_path=tmp_path,
        )

    monkeypatch.setattr("pedantic_parakeet.sources.download_url_source", fake_download)

    source = ResolvedSource(
        source_kind="url",
        original_input="https://example.com/watch?v=123",
        display_name="Bonjour le monde",
        output_stem="Bonjour le monde",
        media_path=None,
    )

    materialized = materialize_source(source)

    assert materialized.media_path == downloaded_file
    assert materialized.is_temporary is True
    assert materialized.cleanup_path == tmp_path


def test_cleanup_materialized_source_removes_temporary_downloads(tmp_path: Path):
    downloaded_file = tmp_path / "downloaded.webm"
    downloaded_file.write_text("x", encoding="utf-8")
    materialized = MaterializedSource(
        display_name="downloaded.webm",
        output_stem="downloaded",
        media_path=downloaded_file,
        is_temporary=True,
        cleanup_path=tmp_path,
    )

    cleanup_materialized_source(materialized)

    assert not tmp_path.exists()
