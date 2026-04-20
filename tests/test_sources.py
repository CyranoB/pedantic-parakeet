"""Tests for media source resolution and URL acquisition."""

from pathlib import Path

import pytest

from pedantic_parakeet.sources import (
    InputResolutionError,
    MaterializedSource,
    ResolvedSource,
    _pick_downloaded_media,
    cleanup_materialized_source,
    is_url_input,
    materialize_source,
    resolve_inputs,
)


def make_resolved_source(
    source_kind: str,
    original_input: str,
    display_name: str,
    output_stem: str,
    media_path: Path | None = None,
) -> ResolvedSource:
    return ResolvedSource(
        source_kind=source_kind,
        original_input=original_input,
        display_name=display_name,
        output_stem=output_stem,
        media_path=media_path,
    )


def make_materialized_source(
    display_name: str,
    output_stem: str,
    media_path: Path,
    is_temporary: bool,
    cleanup_path: Path | None,
) -> MaterializedSource:
    return MaterializedSource(
        display_name=display_name,
        output_stem=output_stem,
        media_path=media_path,
        is_temporary=is_temporary,
        cleanup_path=cleanup_path,
    )


def test_is_url_input_recognizes_http_and_https():
    assert is_url_input("https://example.com/video")
    assert is_url_input("https://example.com/watch?v=123")
    assert not is_url_input("/Users/example/video.mp4")
    assert not is_url_input("recording.m4a")


def test_is_url_input_rejects_private_and_link_local_hosts():
    assert not is_url_input("http://127.0.0.1/video")
    assert not is_url_input("http://192.168.1.20/video")
    assert not is_url_input("http://169.254.169.254/latest/meta-data")
    assert not is_url_input("http://[::1]/video")


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


def test_resolve_inputs_raises_for_missing_local_paths(tmp_path: Path):
    missing_file = tmp_path / "does-not-exist.mp4"
    with pytest.raises(InputResolutionError, match="Input does not exist"):
        resolve_inputs([str(missing_file)])


def test_resolve_inputs_uses_metadata_for_public_urls(monkeypatch: pytest.MonkeyPatch):
    def fake_fetch(url: str) -> dict[str, str]:
        assert url == "https://example.com/watch?v=123"
        return {"title": "Bonjour le monde", "id": "123"}

    monkeypatch.setattr("pedantic_parakeet.sources.fetch_url_metadata", fake_fetch)

    sources = resolve_inputs(["https://example.com/watch?v=123"])

    assert sources == [
        make_resolved_source(
            source_kind="url",
            original_input="https://example.com/watch?v=123",
            display_name="Bonjour le monde",
            output_stem="Bonjour le monde",
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
    source = make_resolved_source(
        source_kind="local",
        original_input=str(media_path),
        display_name=media_path.name,
        output_stem=media_path.stem,
        media_path=media_path,
    )

    materialized = materialize_source(source)

    assert materialized == make_materialized_source(
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
        return make_materialized_source(
            display_name=source.display_name,
            output_stem=source.output_stem,
            media_path=downloaded_file,
            is_temporary=True,
            cleanup_path=tmp_path,
        )

    monkeypatch.setattr("pedantic_parakeet.sources.download_url_source", fake_download)

    source = make_resolved_source(
        source_kind="url",
        original_input="https://example.com/watch?v=123",
        display_name="Bonjour le monde",
        output_stem="Bonjour le monde",
    )

    materialized = materialize_source(source)

    assert materialized.media_path == downloaded_file
    assert materialized.is_temporary is True
    assert materialized.cleanup_path == tmp_path


def test_cleanup_materialized_source_removes_temporary_downloads(tmp_path: Path):
    downloaded_file = tmp_path / "downloaded.webm"
    downloaded_file.write_text("x", encoding="utf-8")
    materialized = make_materialized_source(
        display_name="downloaded.webm",
        output_stem="downloaded",
        media_path=downloaded_file,
        is_temporary=True,
        cleanup_path=tmp_path,
    )

    cleanup_materialized_source(materialized)

    assert not tmp_path.exists()


def test_pick_downloaded_media_prefers_largest_supported_media(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
):
    smaller_media = tmp_path / "small.m4a"
    smaller_media.write_bytes(b"a" * 4)
    larger_media = tmp_path / "large.webm"
    larger_media.write_bytes(b"b" * 8)
    sidecar = tmp_path / "metadata.json"
    sidecar.write_text("{}", encoding="utf-8")

    selected = _pick_downloaded_media(tmp_path)

    assert selected == larger_media
    assert "Multiple media files" in str(recwarn.pop(UserWarning).message)
