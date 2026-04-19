"""Tests for CLI input resolution and model defaults."""

from pathlib import Path

from typer.testing import CliRunner

from pedantic_parakeet.types import Segment, Token, TranscriptionResult

runner = CliRunner()


def make_result(audio_path: str) -> TranscriptionResult:
    return TranscriptionResult(
        text="hello world",
        segments=[
            Segment(
                text="hello world",
                start=0.0,
                end=1.0,
                confidence=0.99,
                tokens=[
                    Token(text="hello", start=0.0, end=0.5, confidence=0.99),
                    Token(text="world", start=0.5, end=1.0, confidence=0.99),
                ],
            )
        ],
        audio_path=audio_path,
        model_id="whisper",
    )


class DummyTranscriber:
    """Minimal stub used to capture CLI model selection."""

    model_id_seen: str | None = None

    def __init__(self, model_id, backend, chunk_duration, language, language_strength):
        type(self).model_id_seen = model_id

    def transcribe(self, audio_path):
        return make_result(str(audio_path))


def test_dry_run_accepts_public_url(monkeypatch):
    from pedantic_parakeet import cli
    from pedantic_parakeet.sources import ResolvedSource

    monkeypatch.setattr(
        cli,
        "resolve_inputs",
        lambda inputs, recursive: [
            ResolvedSource(
                source_kind="url",
                original_input=inputs[0],
                display_name="Bonjour le monde",
                output_stem="Bonjour le monde",
                media_path=None,
            )
        ],
    )

    result = runner.invoke(cli.app, ["--dry-run", "https://example.com/watch?v=123"])

    assert result.exit_code == 0
    assert "Bonjour le monde" in result.output
    assert ".srt" in result.output


def test_cli_uses_whisper_as_default_model(tmp_path: Path, monkeypatch):
    from pedantic_parakeet import cli
    from pedantic_parakeet.sources import MaterializedSource, ResolvedSource

    media_file = tmp_path / "sample.wav"
    media_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "resolve_inputs",
        lambda inputs, recursive: [
            ResolvedSource(
                source_kind="local",
                original_input=inputs[0],
                display_name=media_file.name,
                output_stem=media_file.stem,
                media_path=media_file,
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "materialize_source",
        lambda source: MaterializedSource(
            display_name=source.display_name,
            output_stem=source.output_stem,
            media_path=media_file,
            is_temporary=False,
            cleanup_path=None,
        ),
    )
    monkeypatch.setattr(cli, "cleanup_materialized_source", lambda source: None)
    monkeypatch.setattr(cli, "check_ffmpeg", lambda: True)
    monkeypatch.setattr(cli, "Transcriber", DummyTranscriber)

    result = runner.invoke(cli.app, [str(media_file), "--format", "txt"])

    assert result.exit_code == 0
    assert DummyTranscriber.model_id_seen == "whisper"


def test_cli_preserves_explicit_model_override(tmp_path: Path, monkeypatch):
    from pedantic_parakeet import cli
    from pedantic_parakeet.sources import MaterializedSource, ResolvedSource

    media_file = tmp_path / "sample.wav"
    media_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "resolve_inputs",
        lambda inputs, recursive: [
            ResolvedSource(
                source_kind="local",
                original_input=inputs[0],
                display_name=media_file.name,
                output_stem=media_file.stem,
                media_path=media_file,
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "materialize_source",
        lambda source: MaterializedSource(
            display_name=source.display_name,
            output_stem=source.output_stem,
            media_path=media_file,
            is_temporary=False,
            cleanup_path=None,
        ),
    )
    monkeypatch.setattr(cli, "cleanup_materialized_source", lambda source: None)
    monkeypatch.setattr(cli, "check_ffmpeg", lambda: True)
    monkeypatch.setattr(cli, "Transcriber", DummyTranscriber)

    result = runner.invoke(cli.app, [str(media_file), "--model", "parakeet", "--format", "txt"])

    assert result.exit_code == 0
    assert DummyTranscriber.model_id_seen == "parakeet"
