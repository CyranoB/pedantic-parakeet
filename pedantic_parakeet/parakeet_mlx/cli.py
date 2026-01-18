import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import typer
from mlx.core import bfloat16, float32
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from typing_extensions import Annotated

from . import AlignedResult, AlignedSentence, AlignedToken, from_pretrained
from .alignment import SentenceConfig
from .parakeet import Beam, DecodingConfig, Greedy

app = typer.Typer(no_args_is_help=True)


# helpers
def format_timestamp(
    seconds: float, always_include_hours: bool = True, decimal_marker: str = ","
) -> str:
    assert seconds >= 0
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def to_txt(result: AlignedResult) -> str:
    """Format transcription result as plain text."""
    return result.text.strip()


def _format_highlighted_entry(
    sentence: AlignedSentence,
    token_idx: int,
    decimal_marker: str,
    highlight_tag: str,
) -> tuple[str, str, str]:
    """Format a single highlighted word entry for SRT/VTT."""
    token = sentence.tokens[token_idx]
    start_time = format_timestamp(token.start, decimal_marker=decimal_marker)

    # End time is either the token's end or the next token's start
    is_last_token = token_idx == len(sentence.tokens) - 1
    end_token_time = token.end if is_last_token else sentence.tokens[token_idx + 1].start
    end_time = format_timestamp(end_token_time, decimal_marker=decimal_marker)

    # Build text with highlighted current word
    text_parts = []
    for j, inner_token in enumerate(sentence.tokens):
        if token_idx == j:
            highlighted = inner_token.text.replace(
                inner_token.text.strip(),
                f"<{highlight_tag}>{inner_token.text.strip()}</{highlight_tag}>",
            )
            text_parts.append(highlighted)
        else:
            text_parts.append(inner_token.text)

    return start_time, end_time, "".join(text_parts).strip()


def _format_subtitle_with_highlights(
    result: AlignedResult,
    decimal_marker: str,
    highlight_tag: str,
    include_entry_number: bool,
) -> list[str]:
    """Format subtitle entries with word highlighting."""
    content: list[str] = []
    entry_index = 1

    for sentence in result.sentences:
        for i in range(len(sentence.tokens)):
            start_time, end_time, text = _format_highlighted_entry(
                sentence, i, decimal_marker, highlight_tag
            )
            if include_entry_number:
                content.append(f"{entry_index}")
            content.append(f"{start_time} --> {end_time}")
            content.append(text)
            content.append("")
            entry_index += 1

    return content


def _format_subtitle_sentences(
    result: AlignedResult,
    decimal_marker: str,
    include_entry_number: bool,
) -> list[str]:
    """Format subtitle entries at sentence level."""
    content: list[str] = []
    entry_index = 1

    for sentence in result.sentences:
        start_time = format_timestamp(sentence.start, decimal_marker=decimal_marker)
        end_time = format_timestamp(sentence.end, decimal_marker=decimal_marker)
        text = sentence.text.strip()

        if include_entry_number:
            content.append(f"{entry_index}")
        content.append(f"{start_time} --> {end_time}")
        content.append(text)
        content.append("")
        entry_index += 1

    return content


def to_srt(result: AlignedResult, highlight_words: bool = False) -> str:
    """Format transcription result as an SRT file."""
    if highlight_words:
        content = _format_subtitle_with_highlights(
            result, decimal_marker=",", highlight_tag="u", include_entry_number=True
        )
    else:
        content = _format_subtitle_sentences(
            result, decimal_marker=",", include_entry_number=True
        )
    return "\n".join(content)


def to_vtt(result: AlignedResult, highlight_words: bool = False) -> str:
    """Format transcription result as a VTT file."""
    header = ["WEBVTT", ""]
    if highlight_words:
        content = _format_subtitle_with_highlights(
            result, decimal_marker=".", highlight_tag="b", include_entry_number=False
        )
    else:
        content = _format_subtitle_sentences(
            result, decimal_marker=".", include_entry_number=False
        )
    return "\n".join(header + content)


def _aligned_token_to_dict(token: AlignedToken) -> Dict[str, Any]:
    return {
        "text": token.text,
        "start": round(token.start, 3),
        "end": round(token.end, 3),
        "duration": round(token.duration, 3),
        "confidence": round(token.confidence, 3),
    }


def _aligned_sentence_to_dict(sentence: AlignedSentence) -> Dict[str, Any]:
    return {
        "text": sentence.text,
        "start": round(sentence.start, 3),
        "end": round(sentence.end, 3),
        "duration": round(sentence.duration, 3),
        "confidence": round(sentence.confidence, 3),
        "tokens": [_aligned_token_to_dict(token) for token in sentence.tokens],
    }


def to_json(result: AlignedResult) -> str:
    output_dict = {
        "text": result.text,
        "sentences": [
            _aligned_sentence_to_dict(sentence) for sentence in result.sentences
        ],
    }
    return json.dumps(output_dict, indent=2, ensure_ascii=False)


def _load_model(
    model: str,
    fp32: bool,
    cache_dir: Optional[Path],
    local_attention: bool,
    local_attention_context_size: int,
    verbose: bool,
):
    """Load and configure the transcription model."""
    if verbose:
        print(f"Loading model: [bold cyan]{model}[/bold cyan]...")

    dtype = float32 if fp32 else bfloat16
    loaded_model = from_pretrained(model, dtype=dtype, cache_dir=cache_dir)

    if local_attention:
        loaded_model.encoder.set_attention_model(
            "rel_pos_local_attn",
            (local_attention_context_size, local_attention_context_size),
        )

    if verbose:
        print("[green]Model loaded successfully.[/green]")

    return loaded_model


def _get_formatters(highlight_words: bool) -> Dict[str, Any]:
    """Return dictionary of output format functions."""
    return {
        "txt": to_txt,
        "srt": lambda r: to_srt(r, highlight_words=highlight_words),
        "vtt": lambda r: to_vtt(r, highlight_words=highlight_words),
        "json": to_json,
    }


def _resolve_output_formats(output_format: str, formatters: Dict[str, Any]) -> List[str]:
    """Resolve output format string to list of formats."""
    if output_format == "all":
        return list(formatters.keys())
    if output_format in formatters:
        return [output_format]
    raise ValueError(
        f"Invalid output format '{output_format}'. Choose from {list(formatters.keys()) + ['all']}."
    )


def _print_verbose_result(result: AlignedResult) -> None:
    """Print transcription result in verbose mode."""
    for sentence in result.sentences:
        conf_str = f" [dim](confidence: {sentence.confidence:.2%})[/dim]"
        timestamps = f"[{format_timestamp(sentence.start)} --> {format_timestamp(sentence.end)}]"
        print(f"[blue]{timestamps}[/blue]{conf_str} {sentence.text.strip()}")


def _save_output_files(
    result: AlignedResult,
    audio_path: Path,
    output_dir: Path,
    output_template: str,
    file_index: int,
    formats_to_generate: List[str],
    formatters: Dict[str, Any],
    verbose: bool,
) -> None:
    """Save transcription result to output files."""
    template_vars = {
        "filename": audio_path.stem,
        "parent": str(audio_path.parent),
        "date": datetime.datetime.now().strftime("%Y%m%d"),
        "index": str(file_index + 1),
    }
    output_basename = output_template.format(**template_vars)

    for fmt in formats_to_generate:
        output_content = formatters[fmt](result)
        output_filename = Path(f"{output_basename}.{fmt}")
        output_filepath = output_filename if output_filename.is_absolute() else output_dir / output_filename

        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(output_content)
            if verbose:
                print(f"[green]Saved {fmt.upper()}:[/green] {output_filepath.absolute()}")
        except Exception as e:
            print(f"[bold red]Error writing output file {output_filepath}:[/bold red] {e}")


def _create_decoding_config(
    decoding: str,
    beam_size: int,
    length_penalty: float,
    patience: float,
    duration_reward: float,
    max_words: Optional[int],
    silence_gap: Optional[float],
    max_duration: Optional[float],
) -> DecodingConfig:
    """Create decoding configuration from CLI parameters."""
    decoding_method = (
        Beam(
            beam_size=beam_size,
            length_penalty=length_penalty,
            patience=patience,
            duration_reward=duration_reward,
        )
        if decoding == "beam"
        else Greedy()
    )
    return DecodingConfig(
        decoding=decoding_method,
        sentence=SentenceConfig(
            max_words=max_words, silence_gap=silence_gap, max_duration=max_duration
        ),
    )


def _print_verbose_setup(
    output_dir: Path,
    output_format: str,
    highlight_words: bool,
    total_files: int,
) -> None:
    """Print verbose setup information."""
    print(f"Output directory: [bold cyan]{output_dir.resolve()}[/bold cyan]")
    print(f"Output format(s): [bold cyan]{output_format}[/bold cyan]")
    if output_format in ["srt", "vtt", "all"] and highlight_words:
        print("Highlight words: [bold cyan]Enabled[/bold cyan]")
    print(f"Transcribing {total_files} file(s)...")


def _process_single_audio(
    audio_path: Path,
    loaded_model,
    dtype,
    chunk_duration: Optional[float],
    overlap_duration: float,
    decoding_config: DecodingConfig,
    progress,
    task,
    file_index: int,
    total_files: int,
    verbose: bool,
) -> AlignedResult:
    """Process a single audio file and return the result."""
    if verbose:
        print(f"\nProcessing file {file_index + 1}/{total_files}: [bold cyan]{audio_path.name}[/bold cyan]")
    else:
        progress.update(task, description=f"Processing [cyan]{audio_path.name}[/cyan]...")

    return loaded_model.transcribe(
        audio_path,
        dtype=dtype,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
        chunk_callback=lambda current, full, i=file_index: progress.update(
            task, total=total_files * full, completed=full * i + current
        ),
        decoding_config=decoding_config,
    )


def _process_audio_files(
    audios: List[Path],
    loaded_model,
    dtype,
    chunk_duration: Optional[float],
    overlap_duration: float,
    decoding_config: DecodingConfig,
    output_dir: Path,
    output_template: str,
    formats_to_generate: List[str],
    formatters: Dict[str, Any],
    verbose: bool,
) -> None:
    """Process all audio files with progress tracking."""
    total_files = len(audios)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Transcribing...", total=total_files)

        for i, audio_path in enumerate(audios):
            try:
                result = _process_single_audio(
                    audio_path, loaded_model, dtype, chunk_duration,
                    overlap_duration, decoding_config, progress, task,
                    i, total_files, verbose
                )

                if verbose:
                    _print_verbose_result(result)

                _save_output_files(
                    result, audio_path, output_dir, output_template,
                    i, formats_to_generate, formatters, verbose
                )

            except Exception as e:
                print(f"[bold red]Error transcribing file {audio_path}:[/bold red] {e}")

            progress.update(task, total=total_files, completed=i + 1)


@app.command("transcribe")
def transcribe(
    audios: Annotated[
        List[Path],
        typer.Argument(
            help="Files to transcribe",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            help="Hugging Face repository of model to use", envvar="PARAKEET_MODEL"
        ),
    ] = "mlx-community/parakeet-tdt-0.6b-v3",
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save transcriptions")
    ] = Path("."),
    output_format: Annotated[
        str,
        typer.Option(
            help="Format for output files (txt, srt, vtt, json, all)",
            envvar="PARAKEET_OUTPUT_FORMAT",
        ),
    ] = "srt",
    output_template: Annotated[
        str,
        typer.Option(
            help="Template for output filenames, e.g. '{parent}/{filename}_{date}_{index}'",
            envvar="PARAKEET_OUTPUT_TEMPLATE",
        ),
    ] = "{filename}",
    highlight_words: Annotated[
        bool,
        typer.Option(help="Underline/timestamp each word as it is spoken in srt/vtt"),
    ] = False,
    decoding: Annotated[
        Literal["greedy", "beam"],
        typer.Option(
            help="Decoding method to use",
            envvar="PARAKEET_DECODING",
        ),
    ] = "greedy",
    chunk_duration: Annotated[
        float,
        typer.Option(
            help="Chunking duration in seconds for long audio, 0 to disable chunking.",
            envvar="PARAKEET_CHUNK_DURATION",
        ),
    ] = 60 * 2,
    overlap_duration: Annotated[
        float,
        typer.Option(
            help="Overlap duration in seconds if using chunking",
            envvar="PARAKEET_OVERLAP_DURATION",
        ),
    ] = 15,
    beam_size: Annotated[
        int,
        typer.Option(
            help="Beam size (only used while beam decoding)",
            envvar="PARAKEET_BEAM_SIZE",
        ),
    ] = 5,
    length_penalty: Annotated[
        float,
        typer.Option(
            help="Length penalty in beam. 0.0 to disable (only used while beam decoding)",
            envvar="PARAKEET_LENGTH_PENALTY",
        ),
    ] = 0.013,
    patience: Annotated[
        float,
        typer.Option(
            help="Patience in beam. 1.0 to disable (only used while beam decoding)",
            envvar="PARAKEET_PATIENCE",
        ),
    ] = 3.5,
    duration_reward: Annotated[
        float,
        typer.Option(
            help="From 0.0 to 1.0, < 0.5 to favor token logprobs more, > 0.5 to favor duration logprobs more. (only used while beam decoding in TDT)",
            envvar="PARAKEET_DURATION_REWARD",
        ),
    ] = 0.67,
    max_words: Annotated[
        Optional[int],
        typer.Option(
            "--max-words",
            help="Max words per sentence",
            envvar="PARAKEET_MAX_WORDS",
        ),
    ] = None,
    silence_gap: Annotated[
        Optional[float],
        typer.Option(
            "--silence-gap",
            help="Split at silence gaps (seconds)",
            envvar="PARAKEET_SILENCE_GAP",
        ),
    ] = None,
    max_duration: Annotated[
        Optional[float],
        typer.Option(
            "--max-duration",
            help="Max sentence duration (seconds)",
            envvar="PARAKEET_MAX_DURATION",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print out process and debug messages"),
    ] = False,
    fp32: Annotated[
        bool,
        typer.Option(
            "--fp32/--bf16", help="Use FP32 precision", envvar="PARAKEET_FP32"
        ),
    ] = False,
    local_attention: Annotated[
        bool,
        typer.Option(
            "--local-attention/--full-attention",
            help="Use local attention (reduces intermediate memory usage for long audio)",
            envvar="PARAKEET_LOCAL_ATTENTION",
        ),
    ] = False,
    local_attention_context_size: Annotated[
        int,
        typer.Option(
            "--local-attention-context-size",
            help="Local attention context size (Only applies if using local attention)",
            envvar="PARAKEET_LOCAL_ATTENTION_CTX",
        ),
    ] = 256,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory for HuggingFace model cache. If not specified, uses HF's default cache location",
            envvar="PARAKEET_CACHE_DIR",
        ),
    ] = None,
):
    """
    Transcribe audio files using Parakeet MLX models.
    """
    try:
        loaded_model = _load_model(
            model, fp32, cache_dir, local_attention, local_attention_context_size, verbose
        )
    except Exception as e:
        print(f"[bold red]Error loading model {model}:[/bold red] {e}")
        raise typer.Exit(code=1)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[bold red]Error creating output directory {output_dir}:[/bold red] {e}")
        raise typer.Exit(code=1)

    formatters = _get_formatters(highlight_words)
    try:
        formats_to_generate = _resolve_output_formats(output_format, formatters)
    except ValueError as e:
        print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)

    if verbose:
        _print_verbose_setup(output_dir, output_format, highlight_words, len(audios))

    decoding_config = _create_decoding_config(
        decoding, beam_size, length_penalty, patience, duration_reward,
        max_words, silence_gap, max_duration
    )
    if verbose:
        print("Decoding config being used:", decoding_config)

    dtype = float32 if fp32 else bfloat16
    chunk_dur = chunk_duration if chunk_duration != 0 else None

    _process_audio_files(
        audios, loaded_model, dtype, chunk_dur, overlap_duration, decoding_config,
        output_dir, output_template, formats_to_generate, formatters, verbose
    )

    print(
        f"\n[bold green]{model.removeprefix('mlx-community/')} transcription complete.[/bold green] Outputs saved in '{output_dir.resolve()}'."
    )


if __name__ == "__main__":
    app()
