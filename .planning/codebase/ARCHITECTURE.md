# Architecture

**Analysis Date:** 2026-01-19

## Pattern Overview

**Overall:** Facade + Strategy Pattern with Plugin Registry

**Key Characteristics:**
- CLI delegates to `Transcriber` facade for all transcription operations
- `Transcriber` routes to backend implementations via registry-based model resolution
- Backend implementations follow a common `BaseTranscriber` protocol
- Data types shared across layers via dedicated types module
- Vendored dependencies (`parakeet_mlx`) for core ML model support

## Layers

**CLI Layer (`pedantic_parakeet/cli.py`):**
- Purpose: User interface, argument parsing, output formatting
- Location: `pedantic_parakeet/cli.py`
- Contains: Typer CLI app, validation functions, progress display, file I/O
- Depends on: Transcriber facade, formatters, audio discovery, backends registry
- Used by: End users via `pedantic-parakeet` command

**Facade Layer (`pedantic_parakeet/transcriber.py`):**
- Purpose: Unified API for transcription, backend selection, lazy loading
- Location: `pedantic_parakeet/transcriber.py`
- Contains: `Transcriber` class, `create_backend()` factory function
- Depends on: Backend implementations, types, registry
- Used by: CLI layer

**Backend Layer (`pedantic_parakeet/backends/`):**
- Purpose: STT model implementations with common interface
- Location: `pedantic_parakeet/backends/`
- Contains: `ParakeetBackend`, `MlxAudioBackend`, registry, base contracts
- Depends on: Vendored parakeet_mlx, optional mlx-audio library, types
- Used by: Transcriber facade

**Core ML Layer (`pedantic_parakeet/parakeet_mlx/`):**
- Purpose: Vendored Parakeet TDT model implementation
- Location: `pedantic_parakeet/parakeet_mlx/`
- Contains: Conformer encoder, attention, alignment, audio processing
- Depends on: MLX framework, HuggingFace Hub, librosa
- Used by: ParakeetBackend

**Utilities Layer:**
- Purpose: Cross-cutting concerns
- Location: `pedantic_parakeet/audio.py`, `pedantic_parakeet/formatters.py`, `pedantic_parakeet/language_bias.py`
- Contains: Audio file discovery, output formatting (SRT/VTT/JSON/TXT), language bias vectors
- Depends on: Types, external tools (ffmpeg)
- Used by: CLI layer, backends

## Data Flow

**Transcription Request Flow:**

1. CLI receives audio path(s) and options via Typer
2. CLI validates inputs (formats, language, model capabilities)
3. CLI creates `Transcriber` with resolved model/backend
4. `Transcriber.transcribe()` lazy-loads backend on first call
5. Backend loads ML model (via HuggingFace Hub cache)
6. Backend processes audio in chunks (if enabled)
7. Backend returns `TranscriptionResult` with segments
8. CLI formats output via formatters module
9. CLI writes output files (txt/srt/vtt/json)

**Model Resolution Flow:**

1. User provides model ID or alias (e.g., "parakeet", "whisper")
2. `resolve_model()` looks up in `MODEL_REGISTRY`
3. Returns `ModelInfo` with backend type and capabilities
4. `create_backend()` instantiates correct backend class
5. Backend uses model_id to load from HuggingFace Hub

**State Management:**
- No persistent state; each run is independent
- Model cached in `~/.cache/huggingface/` by HuggingFace Hub
- Backend instance cached in Transcriber for reuse across files

## Key Abstractions

**BaseTranscriber Protocol (`pedantic_parakeet/backends/base.py`):**
- Purpose: Contract for all transcription backends
- Examples: `pedantic_parakeet/backends/parakeet.py`, `pedantic_parakeet/backends/mlx_audio.py`
- Pattern: Protocol (structural typing) with required methods: `transcribe()`, `model_id`, `capabilities`

**TranscriptionResult (`pedantic_parakeet/types.py`):**
- Purpose: Unified output format for all backends
- Examples: Used by formatters, returned from backends
- Pattern: Dataclass with nested Segment/Token structures

**STTCapabilities (`pedantic_parakeet/backends/base.py`):**
- Purpose: Declare feature support per model/backend
- Examples: `supports_timestamps`, `supports_language_bias`, `supports_language_hint`
- Pattern: Frozen dataclass for immutable capability flags

**ModelInfo (`pedantic_parakeet/backends/base.py`):**
- Purpose: Metadata for curated models (model_id, backend, capabilities, aliases)
- Examples: Registry entries in `pedantic_parakeet/backends/registry.py`
- Pattern: Dataclass with optional fields

## Entry Points

**CLI Entry Point:**
- Location: `pedantic_parakeet/cli.py:app`
- Triggers: `pedantic-parakeet` command (defined in `pyproject.toml`)
- Responsibilities: Parse args, validate inputs, orchestrate transcription, write outputs

**Programmatic Entry Point:**
- Location: `pedantic_parakeet/transcriber.py:Transcriber`
- Triggers: Direct Python import and instantiation
- Responsibilities: Provide high-level transcription API

**Backend Factory:**
- Location: `pedantic_parakeet/transcriber.py:create_backend()`
- Triggers: Called by Transcriber or directly
- Responsibilities: Instantiate correct backend based on model

## Error Handling

**Strategy:** Fail-fast with helpful error messages

**Patterns:**
- CLI validates all inputs before processing (formats, language, model capabilities)
- `typer.BadParameter` for user input errors with param hints
- `ValueError` for programmatic API errors with context
- `RuntimeError` for missing optional dependencies (mlx-audio)
- Per-file error handling with `--fail-fast` / `--continue-on-error` options

## Cross-Cutting Concerns

**Logging:** No dedicated logging framework; uses Rich console for user feedback

**Validation:** 
- CLI layer validates user inputs via `_validate_format_capabilities()`, `_validate_language_capabilities()`
- Registry provides capability checks before backend instantiation

**Model Loading:**
- Lazy loading: Models loaded on first `transcribe()` call
- Caching: HuggingFace Hub handles model file caching
- Progress: Rich progress bars for multi-file processing

---

*Architecture analysis: 2026-01-19*
