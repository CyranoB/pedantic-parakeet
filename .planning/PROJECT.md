# Project: Pedantic Parakeet

## Core Value

CLI transcription tool that produces high-quality, multi-format transcripts from audio files using MLX models optimized for Apple Silicon.

## Project Description

Pedantic Parakeet is a command-line tool for transcribing audio files. It wraps NVIDIA Parakeet TDT models via MLX for Apple Silicon, with support for additional backends (mlx-audio for Whisper, Voxtral). Key features include:

- Batch processing of audio files and directories
- Multiple output formats (txt, srt, vtt, json)
- Language bias to reduce code-switching for non-English content
- Curated model registry with capability-based validation

## Constraints

- **Platform:** Apple Silicon (MLX-based)
- **Dependency:** ffmpeg required for audio conversion
- **Optional:** mlx-audio package for additional models
- **Languages:** Language bias currently supports French only

## Key Decisions

| Date | Decision | Rationale | Phase |
|------|----------|-----------|-------|
| 2026-01-19 | Vendored parakeet-mlx | Enable language bias modifications without upstream changes | Pre-v1 |
| 2026-01-20 | Protocol-based backend interface | Duck-typing over inheritance for flexibility | 01 |
| 2026-01-20 | Extract types to types.py | Resolve circular imports between transcriber and backends | 01 |
| 2026-01-20 | Model aliases in registry | CLI convenience (--model whisper instead of full HF ID) | 01 |
| 2026-01-20 | Lazy mlx-audio import | Allow module import without optional dependency | 01 |

## Tech Stack

- **Runtime:** Python 3.10+
- **ML Framework:** MLX
- **CLI Framework:** Typer + Rich
- **Audio Processing:** librosa, ffmpeg
- **Build:** Hatchling

## Repository Structure

```
pedantic_parakeet/
  cli.py              # CLI entry point
  transcriber.py      # Facade with backend factory
  types.py            # Shared dataclasses (Token, Segment, TranscriptionResult)
  formatters.py       # Output format generators
  audio.py            # Audio discovery utilities
  language_bias.py    # French language bias support
  backends/           # Backend implementations
    base.py           # Contracts (Backend enum, STTCapabilities, ModelInfo)
    registry.py       # Curated model registry
    parakeet.py       # Parakeet backend (vendored)
    mlx_audio.py      # mlx-audio backend
  parakeet_mlx/       # Vendored parakeet-mlx package
tests/
  test_cli.py         # CLI behavior tests
  test_backends.py    # Registry and capability tests
```

---

*Created: 2026-01-20*
