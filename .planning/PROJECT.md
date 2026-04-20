# Project: Pedantic Parakeet

## Current State

**Version:** v1.0 MVP (shipped 2026-01-20)
**Status:** Production-ready

Pedantic Parakeet is a CLI transcription tool for Apple Silicon with multi-backend support (Parakeet, Whisper, Voxtral), capability-based validation, and comprehensive test coverage.

## Core Value

CLI transcription tool that produces high-quality, multi-format transcripts from audio files using MLX models optimized for Apple Silicon.

## Project Description

Pedantic Parakeet is a command-line tool for transcribing audio files. It wraps NVIDIA Parakeet TDT models via MLX for Apple Silicon, with support for additional backends (mlx-audio for Whisper, Voxtral). Key features include:

- Batch processing of audio files and directories
- Multiple output formats (txt, srt, vtt, json)
- Language bias to reduce code-switching for non-English content
- Curated model registry with capability-based validation

## Requirements

### Validated (v1.0)

- ✓ Multi-backend support (Parakeet, mlx-audio) — v1.0
- ✓ Model listing with `--list-models` — v1.0
- ✓ Backend/model selection via CLI — v1.0
- ✓ Format validation for text-only models — v1.0
- ✓ Language bias validation — v1.0
- ✓ 145 tests covering registry and CLI — v1.0

### Active

(None — run `/gsd-new-milestone` to define v1.1 requirements)

### Out of Scope

- Mobile app — web-first approach
- GUI application — CLI focus
- Real-time streaming — batch processing focus

## Constraints

- **Platform:** Apple Silicon (MLX-based)
- **Dependency:** ffmpeg required for audio conversion
- **Optional:** mlx-audio package for additional models
- **Languages:** Language bias currently supports French only

## Key Decisions

| Date | Decision | Rationale | Phase | Outcome |
|------|----------|-----------|-------|---------|
| 2026-01-19 | Vendored parakeet-mlx | Enable language bias modifications without upstream changes | Pre-v1 | ✓ Good |
| 2026-01-20 | Protocol-based backend interface | Duck-typing over inheritance for flexibility | 01 | ✓ Good |
| 2026-01-20 | Extract types to types.py | Resolve circular imports between transcriber and backends | 01 | ✓ Good |
| 2026-01-20 | Model aliases in registry | CLI convenience (--model whisper instead of full HF ID) | 01 | ✓ Good |
| 2026-01-20 | Lazy mlx-audio import | Allow module import without optional dependency | 01 | ✓ Good |

## Context

**Codebase:**
- ~1,700 LOC Python (excluding vendored parakeet_mlx)
- 145 tests passing
- 4 curated models in registry

**Tech Stack:**
- Python 3.10+
- MLX framework
- Typer + Rich for CLI
- librosa + ffmpeg for audio

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

*Last updated: 2026-01-20 after v1.0 milestone*
