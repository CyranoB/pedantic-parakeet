"""CLI interface for transcription tool."""

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from . import __version__
from .audio import discover_audio_files, check_ffmpeg, SUPPORTED_EXTENSIONS
from .formatters import FORMATTERS, EXTENSIONS, format_txt
from .transcriber import Transcriber, DEFAULT_MODEL

app = typer.Typer(
    name="transcribe",
    help="Transcribe audio files using Parakeet TDT models.",
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


@app.command()
def main(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            help="Audio files or directories to transcribe",
            exists=True,
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
            help="HuggingFace model ID",
        ),
    ] = DEFAULT_MODEL,
    chunk_duration: Annotated[
        float,
        typer.Option(
            "--chunk-duration",
            help="Chunk duration in seconds for long audio (0 to disable)",
        ),
    ] = 120.0,
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
    """Transcribe audio files to text, SRT, VTT, or JSON."""
    # Parse formats
    formats = parse_formats(format)

    # Discover audio files
    audio_files = discover_audio_files(inputs, recursive=recursive)

    if not audio_files:
        err_console.print("[red]No audio files found.[/red]")
        err_console.print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        raise typer.Exit(1)

    # Create output directory if specified
    if output:
        output.mkdir(parents=True, exist_ok=True)

    # Dry run: just show what would be processed
    if dry_run:
        console.print(f"[bold]Would process {len(audio_files)} file(s):[/bold]")
        for audio_path in audio_files:
            out_dir = output or audio_path.parent
            for fmt in formats:
                out_file = out_dir / (audio_path.stem + EXTENSIONS[fmt])
                console.print(f"  {audio_path} → {out_file}")
        raise typer.Exit(0)

    # Check ffmpeg (warning only)
    if not check_ffmpeg():
        err_console.print(
            "[yellow]Warning: ffmpeg not found. Some audio formats may not work.[/yellow]"
        )
        err_console.print("[yellow]Install with: brew install ffmpeg[/yellow]")

    # Initialize transcriber (loads model)
    if verbose:
        console.print(f"[dim]Loading model: {model}...[/dim]")

    transcriber = Transcriber(model_id=model, chunk_duration=chunk_duration)

    # Process files
    success_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        disable=not verbose and len(audio_files) == 1,
    ) as progress:
        task = progress.add_task("Transcribing...", total=len(audio_files))

        for audio_path in audio_files:
            progress.update(task, description=f"[cyan]{audio_path.name}[/cyan]")

            try:
                # Transcribe
                result = transcriber.transcribe(audio_path)

                # Determine output directory
                out_dir = output or audio_path.parent

                # Write outputs
                for fmt in formats:
                    out_file = out_dir / (audio_path.stem + EXTENSIONS[fmt])

                    if fmt == "txt":
                        content = format_txt(result, timestamps=timestamps)
                    else:
                        formatter = FORMATTERS[fmt]
                        content = formatter(result)

                    out_file.write_text(content, encoding="utf-8")

                    if verbose:
                        console.print(f"  [green]✓[/green] {out_file}")

                success_count += 1

            except Exception as e:
                error_count += 1
                err_console.print(f"[red]Error processing {audio_path}: {e}[/red]")

                if fail_fast:
                    raise typer.Exit(1)

            progress.advance(task)

    # Summary
    if len(audio_files) > 1 or verbose:
        console.print()
        console.print(
            f"[bold green]✓ {success_count} file(s) transcribed[/bold green]"
            + (f", [bold red]{error_count} error(s)[/bold red]" if error_count else "")
        )

    if error_count and not fail_fast:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
