"""CLI interface for transcription tool."""

import math
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from . import __version__
from .audio import SUPPORTED_EXTENSIONS, check_ffmpeg
from .backends.base import Backend
from .backends.mlx_audio import is_mlx_audio_available
from .backends.registry import list_models, resolve_model
from .formatters import EXTENSIONS, FORMATTERS, format_txt
from .language_bias import SUPPORTED_LANGUAGES
from .sources import (
    InputResolutionError,
    MaterializedSource,
    ResolvedSource,
    cleanup_materialized_source,
    materialize_source,
    resolve_inputs,
)
from .transcriber import Transcriber

CLI_DEFAULT_MODEL = "whisper"
DEFAULT_LANGUAGE_STRENGTH = 0.5

app = typer.Typer(
    name="transcribe",
    help="Transcribe local media files, directories, or public URLs.",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def parse_formats(format_str: str) -> list[str]:
    """Parse comma-separated format string into list of formats."""
    formats = []
    for part in format_str.split(","):
        fmt = part.strip().lower()
        if fmt == "all":
            return list(FORMATTERS.keys())
        if fmt and fmt in FORMATTERS:
            formats.append(fmt)
        elif fmt:
            err_console.print(f"[yellow]Warning: Unknown format '{fmt}', ignoring[/yellow]")
    return formats or ["srt"]  # Default to srt if nothing valid


def version_callback(value: bool) -> None:
    if value:
        console.print(f"transcribe {__version__}")
        raise typer.Exit()


def _is_model_available(model, mlx_audio_available: bool) -> bool:
    """Return whether the model can be used in the current environment."""
    return model.backend == Backend.PARAKEET or mlx_audio_available


def _format_timestamps_support(supports_timestamps: bool, rich: bool = True) -> str:
    """Format a timestamps support indicator for model listings."""
    if rich:
        return "[green]✓[/green]" if supports_timestamps else "[red]✗[/red]"
    return "✓" if supports_timestamps else "✗"


def _print_available_models(models: list) -> None:
    """Print models that are currently usable."""
    console.print("[bold]Available Models:[/bold]\n")
    for model in models:
        console.print(f"  [cyan]{model.model_id}[/cyan]")
        console.print(f"    Backend: {model.backend}")
        console.print(
            "    Timestamps: "
            f"{_format_timestamps_support(model.capabilities.supports_timestamps)}"
        )
        if model.aliases:
            console.print(f"    Aliases: {', '.join(model.aliases)}")
        console.print(f"    {model.description}")
        console.print()


def _print_unavailable_models(models: list) -> None:
    """Print models that require optional mlx-audio support."""
    if not models:
        return

    console.print("[dim]─" * 50 + "[/dim]")
    console.print("\n[bold dim]Additional Models (requires mlx-audio):[/bold dim]\n")
    console.print("[dim]Install with: pip install 'pedantic-parakeet\\[mlx-audio]'[/dim]\n")
    for model in models:
        aliases = f" ({', '.join(model.aliases)})" if model.aliases else ""
        console.print(f"  [dim]{model.model_id}{aliases}[/dim]")
    console.print()


def list_models_callback(value: bool) -> None:
    """Print curated models and exit."""
    if value:
        mlx_audio_available = is_mlx_audio_available()
        models = list_models()
        available_models = [
            model for model in models if _is_model_available(model, mlx_audio_available)
        ]
        unavailable_models = [
            model for model in models if not _is_model_available(model, mlx_audio_available)
        ]

        _print_available_models(available_models)
        _print_unavailable_models(unavailable_models)

        raise typer.Exit()


def _validate_format_capabilities(formats: list[str], model_id: str) -> None:
    """Validate that requested formats are supported by the model.

    Args:
        formats: List of requested output formats.
        model_id: The model ID or alias to check.

    Raises:
        typer.BadParameter: If model doesn't support timestamps but timed formats requested.
    """
    # Formats that require timestamps
    timed_formats = {"srt", "vtt", "json"}
    requested_timed = set(formats) & timed_formats

    if not requested_timed:
        return  # No timed formats requested, no validation needed

    try:
        model_info = resolve_model(model_id)
        if not model_info.capabilities.supports_timestamps:
            raise typer.BadParameter(
                f"Model '{model_id}' does not support timestamps. "
                f"Cannot use formats: {', '.join(sorted(requested_timed))}. "
                f"Use --format txt instead, or choose a different model (see --list-models).",
                param_hint="--format",
            )
    except ValueError:
        # Unknown model - let it pass, Transcriber will handle it
        pass


def _validate_language_capabilities(
    language: str | None,
    language_strength: float,
    model_id: str,
) -> None:
    """Validate that language options are supported by the model.

    Args:
        language: Target language code or None.
        language_strength: Bias strength value.
        model_id: The model ID or alias to check.

    Raises:
        typer.BadParameter: If model doesn't support requested language features.
    """
    uses_default_strength = math.isclose(
        language_strength,
        DEFAULT_LANGUAGE_STRENGTH,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    if language is None and uses_default_strength:
        return  # No language options used

    try:
        model_info = resolve_model(model_id)
        caps = model_info.capabilities

        # Check language_strength with models that don't support language bias
        if not uses_default_strength and not caps.supports_language_bias:
            supported_models = [
                m.model_id for m in list_models()
                if m.capabilities.supports_language_bias
            ]
            raise typer.BadParameter(
                f"Model '{model_id}' does not support --language-strength. "
                f"Models with language bias support: {', '.join(supported_models)}",
                param_hint="--language-strength",
            )

        # Check language option with models that support neither bias nor hint
        if language and not caps.supports_language_bias and not caps.supports_language_hint:
            supported_models = [
                m.model_id for m in list_models()
                if m.capabilities.supports_language_bias or m.capabilities.supports_language_hint
            ]
            raise typer.BadParameter(
                f"Model '{model_id}' does not support --language. "
                f"Models with language support: {', '.join(supported_models)}",
                param_hint="--language",
            )
    except ValueError:
        # Unknown model - let it pass, Transcriber will handle it
        pass


def _validate_backend_availability(model_id: str, backend: str | None) -> None:
    """Validate that required backend is available.

    Args:
        model_id: The model ID or alias to check.
        backend: Explicit backend override, or None for auto-detection.

    Raises:
        typer.BadParameter: If model requires mlx-audio but it's not installed.
    """
    try:
        model_info = resolve_model(model_id)
        requires_mlx_audio = model_info.backend == Backend.MLX_AUDIO
    except ValueError:
        # Unknown model - check if explicit backend is mlx-audio
        requires_mlx_audio = backend == "mlx-audio"

    # Also check explicit backend override
    if backend == "mlx-audio":
        requires_mlx_audio = True

    if requires_mlx_audio and not is_mlx_audio_available():
        raise typer.BadParameter(
            f"Model '{model_id}' requires the mlx-audio backend which is not installed. "
            "Install with: pip install 'pedantic-parakeet[mlx-audio]'",
            param_hint="--model",
        )


def _write_outputs(
    result,
    output_stem: str,
    base_output_dir: Path,
    formats: list[str],
    timestamps: bool,
    console: Console,
    verbose: bool,
) -> None:
    """Write transcription results to files in all requested formats."""
    for fmt in formats:
        out_file = base_output_dir / (output_stem + EXTENSIONS[fmt])

        if fmt == "txt":
            content = format_txt(result, timestamps=timestamps)
        else:
            formatter = FORMATTERS[fmt]
            content = formatter(result)

        out_file.write_text(content, encoding="utf-8")

        if verbose:
            console.print(f"  [green]✓[/green] {out_file}")


def _show_dry_run(
    sources: list[ResolvedSource],
    formats: list[str],
    output: Path | None,
    console: Console,
) -> None:
    """Show what files would be processed in dry run mode."""
    console.print(f"[bold]Would process {len(sources)} input(s):[/bold]")
    for source in sources:
        out_dir = _get_output_dir(source, output)
        for fmt in formats:
            out_file = out_dir / (source.output_stem + EXTENSIONS[fmt])
            console.print(f"  {source.display_name} → {out_file}")


def _check_ffmpeg_warning(err_console: Console) -> None:
    """Print ffmpeg warning if not found."""
    err_console.print(
        "[yellow]Warning: ffmpeg not found. Some audio formats may not work.[/yellow]"
    )
    err_console.print("[yellow]Install with: brew install ffmpeg[/yellow]")


def _run_dry_run(
    inputs: list[str],
    recursive: bool,
    format: str,
    output: Path | None,
    console: Console,
) -> None:
    """Run dry run and exit."""
    formats = parse_formats(format)
    sources = resolve_inputs(inputs, recursive=recursive)

    if not sources:
        err_console = Console(stderr=True)
        err_console.print("[red]No media files found.[/red]")
        err_console.print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        raise typer.Exit(1)

    _show_dry_run(sources, formats, output, console)
    raise typer.Exit(0)


def _get_output_dir(source: ResolvedSource, output: Path | None) -> Path:
    """Determine where transcript files should be written for a source."""
    if output is not None:
        return output
    if source.media_path is not None:
        return source.media_path.parent
    return Path.cwd()


def _process_files(
    sources: list[ResolvedSource],
    transcriber: Transcriber,
    formats: list[str],
    output: Path | None,
    timestamps: bool,
    console: Console,
    err_console: Console,
    verbose: bool,
    fail_fast: bool,
) -> tuple[int, int]:
    """Process all resolved sources. Returns (success_count, error_count)."""
    success_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        disable=not verbose and len(sources) == 1,
    ) as progress:
        task = progress.add_task("Transcribing...", total=len(sources))

        for source in sources:
            progress.update(task, description=f"[cyan]{source.display_name}[/cyan]")

            if _process_file(
                source,
                transcriber,
                formats,
                output,
                timestamps,
                console,
                err_console,
                verbose,
            ):
                success_count += 1
            else:
                error_count += 1
                if fail_fast:
                    raise typer.Exit(1)

            progress.advance(task)

    return success_count, error_count


def _print_summary(
    sources: list[ResolvedSource],
    success_count: int,
    error_count: int,
    verbose: bool,
    console: Console,
) -> None:
    """Print processing summary."""
    if len(sources) > 1 or verbose:
        console.print()
        console.print(
            f"[bold green]✓ {success_count} file(s) transcribed[/bold green]"
            + (f", [bold red]{error_count} error(s)[/bold red]" if error_count else "")
        )


def _process_file(
    source: ResolvedSource,
    transcriber: Transcriber,
    formats: list[str],
    output: Path | None,
    timestamps: bool,
    console: Console,
    err_console: Console,
    verbose: bool,
) -> bool:
    """Process a single audio file. Returns True on success, False on error."""
    materialized: MaterializedSource | None = None
    try:
        materialized = materialize_source(source)
        result = transcriber.transcribe(materialized.media_path)
        _write_outputs(
            result,
            output_stem=source.output_stem,
            base_output_dir=_get_output_dir(source, output),
            formats=formats,
            timestamps=timestamps,
            console=console,
            verbose=verbose,
        )
        return True
    except Exception as e:
        err_console.print(f"[red]Error processing {source.display_name}: {e}[/red]")
        return False
    finally:
        if materialized is not None:
            cleanup_materialized_source(materialized)


@app.command()
def main(  # NOSONAR - Typer entrypoint intentionally exposes the public CLI option surface.
    inputs: Annotated[
        list[str],
        typer.Argument(
            help="Media files, directories, or public URLs to transcribe",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Output directory (default: same as input file)",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format", "-f",
            help="Output format(s): txt, srt, vtt, json, or 'all'. Comma-separated.",
        ),
    ] = "srt",
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive", "-r",
            help="Search directories recursively",
        ),
    ] = False,
    timestamps: Annotated[
        bool,
        typer.Option(
            "--timestamps", "-t",
            help="Include timestamps in plain text output",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be processed without transcribing",
        ),
    ] = False,
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast/--continue-on-error",
            help="Stop on first error vs continue processing",
        ),
    ] = False,
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m",
            help="HuggingFace model ID or alias (see --list-models)",
        ),
    ] = CLI_DEFAULT_MODEL,
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend", "-b",
            help="Backend: parakeet or mlx-audio (auto-detected from model by default)",
        ),
    ] = None,
    list_models_flag: Annotated[
        bool | None,
        typer.Option(
            "--list-models",
            callback=list_models_callback,
            is_eager=True,
            help="List supported models and exit",
        ),
    ] = None,
    chunk_duration: Annotated[
        float,
        typer.Option(
            "--chunk-duration",
            help="Chunk duration in seconds for long audio (0 to disable)",
        ),
    ] = 120.0,
    language: Annotated[
        str | None,
        typer.Option(
            "--language", "-l",
            help="Target language to reduce code-switching (fr)",
        ),
    ] = None,
    language_strength: Annotated[
        float,
        typer.Option(
            "--language-strength",
            help="Bias strength 0.0-2.0 (default 0.5)",
        ),
    ] = DEFAULT_LANGUAGE_STRENGTH,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v",
            help="Show detailed progress",
        ),
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
) -> None:
    """Transcribe media files or public URLs to text, SRT, VTT, or JSON."""
    # Validate language option
    if language and language not in SUPPORTED_LANGUAGES:
        raise typer.BadParameter(
            f"Unsupported language '{language}'. Must be one of: {sorted(SUPPORTED_LANGUAGES)}",
            param_hint="--language",
        )

    # Validate strength range
    if not 0.0 <= language_strength <= 2.0:
        raise typer.BadParameter(
            "Must be between 0.0 and 2.0",
            param_hint="--language-strength",
        )

    # Parse formats
    formats = parse_formats(format)

    # Validate format capabilities BEFORE instantiating backend
    _validate_format_capabilities(formats, model)

    # Validate language capabilities BEFORE instantiating backend
    _validate_language_capabilities(language, language_strength, model)

    # Resolve local paths and public URLs into transcribable sources
    try:
        sources = resolve_inputs(inputs, recursive=recursive)
    except InputResolutionError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not sources:
        err_console.print("[red]No media files found.[/red]")
        err_console.print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        raise typer.Exit(1)

    # Create output directory if specified
    if output:
        output.mkdir(parents=True, exist_ok=True)

    # Dry run: just show what would be processed
    if dry_run:
        _show_dry_run(sources, formats, output, console)
        raise typer.Exit(0)

    # Validate backend availability before instantiating backend
    _validate_backend_availability(model, backend)

    # Check ffmpeg (warning only)
    if not check_ffmpeg():
        _check_ffmpeg_warning(err_console)

    # Parse backend option
    backend_enum: Backend | None = None
    if backend:
        try:
            backend_enum = Backend(backend)
        except ValueError:
            raise typer.BadParameter(
                f"Invalid backend '{backend}'. Must be one of: parakeet, mlx-audio",
                param_hint="--backend",
            )

    # Initialize transcriber (loads model)
    if verbose:
        console.print(f"[dim]Loading model: {model}...[/dim]")

    transcriber = Transcriber(
        model_id=model,
        backend=backend_enum,
        chunk_duration=chunk_duration,
        language=language,
        language_strength=language_strength,
    )

    success_count, error_count = _process_files(
        sources,
        transcriber,
        formats,
        output,
        timestamps,
        console,
        err_console,
        verbose,
        fail_fast,
    )

    _print_summary(sources, success_count, error_count, verbose, console)

    if error_count and not fail_fast:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
