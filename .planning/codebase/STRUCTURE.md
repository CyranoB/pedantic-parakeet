# Codebase Structure

**Analysis Date:** 2026-01-19

## Directory Layout

```
transcription/
├── pedantic_parakeet/       # Main package
│   ├── __init__.py          # Package version
│   ├── cli.py               # CLI entry point (Typer app)
│   ├── transcriber.py       # Transcriber facade + backend factory
│   ├── types.py             # Shared data types (Token, Segment, TranscriptionResult)
│   ├── audio.py             # Audio file discovery utilities
│   ├── formatters.py        # Output formatters (txt, srt, vtt, json)
│   ├── language_bias.py     # Language bias vector builder
│   ├── backends/            # Backend implementations
│   │   ├── __init__.py      # Lazy import for MlxAudioBackend
│   │   ├── base.py          # Backend contracts (Protocol, enums, dataclasses)
│   │   ├── registry.py      # Model registry and resolution
│   │   ├── parakeet.py      # Parakeet TDT backend
│   │   └── mlx_audio.py     # mlx-audio backend (optional)
│   └── parakeet_mlx/        # Vendored parakeet-mlx library
│       ├── __init__.py      # Public API exports
│       ├── parakeet.py      # Core model classes (ParakeetTDT, etc.)
│       ├── conformer.py     # Conformer encoder
│       ├── attention.py     # Attention mechanisms
│       ├── alignment.py     # Word/sentence alignment
│       ├── audio.py         # Audio preprocessing
│       ├── tokenizer.py     # Tokenization
│       ├── rnnt.py          # RNN-T decoder
│       ├── ctc.py           # CTC decoder
│       ├── cache.py         # KV caching
│       ├── utils.py         # Model loading (from_pretrained)
│       └── cli.py           # Standalone CLI (not used)
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_cli.py          # CLI integration tests
│   ├── test_backends.py     # Backend registry tests
│   ├── test_parakeet.py     # Parakeet model tests
│   ├── test_alignment.py    # Alignment algorithm tests
│   └── test_language_bias.py # Language bias tests
├── pyproject.toml           # Project config and dependencies
├── uv.lock                  # Lock file (uv package manager)
├── README.md                # User documentation
├── LICENSE                  # MIT license
└── CHANGELOG.md             # Version history
```

## Directory Purposes

**`pedantic_parakeet/`:**
- Purpose: Main Python package containing all application code
- Contains: CLI, transcriber facade, backends, utilities
- Key files: `cli.py` (entry), `transcriber.py` (facade), `types.py` (shared types)

**`pedantic_parakeet/backends/`:**
- Purpose: Pluggable transcription backend implementations
- Contains: Base contracts, model registry, Parakeet and mlx-audio backends
- Key files: `base.py` (contracts), `registry.py` (model lookup), `parakeet.py` (main backend)

**`pedantic_parakeet/parakeet_mlx/`:**
- Purpose: Vendored fork of parakeet-mlx library for Parakeet TDT models
- Contains: ML model implementation, audio processing, alignment
- Key files: `parakeet.py` (model classes), `utils.py` (model loading), `alignment.py` (word alignment)

**`tests/`:**
- Purpose: Pytest test suite
- Contains: Unit and integration tests for all modules
- Key files: `test_cli.py` (CLI tests), `test_backends.py` (registry tests)

## Key File Locations

**Entry Points:**
- `pedantic_parakeet/cli.py`: CLI application (`app` Typer instance)
- `pyproject.toml`: Defines `pedantic-parakeet` command

**Configuration:**
- `pyproject.toml`: Dependencies, optional extras, build config, ruff settings
- `.gitignore`: Standard Python ignores + .env, .venv, etc.

**Core Logic:**
- `pedantic_parakeet/transcriber.py`: Transcriber facade and backend factory
- `pedantic_parakeet/backends/parakeet.py`: Primary backend implementation
- `pedantic_parakeet/backends/registry.py`: Model registry with capabilities

**Testing:**
- `tests/test_cli.py`: CLI command tests
- `tests/test_backends.py`: Registry and backend tests

**Types/Contracts:**
- `pedantic_parakeet/types.py`: Token, Segment, TranscriptionResult
- `pedantic_parakeet/backends/base.py`: Backend, STTCapabilities, ModelInfo, BaseTranscriber

## Naming Conventions

**Files:**
- snake_case for all Python files: `transcriber.py`, `language_bias.py`
- Test files prefixed with `test_`: `test_cli.py`, `test_backends.py`

**Directories:**
- snake_case for packages: `pedantic_parakeet`, `parakeet_mlx`
- Lowercase plural for collections: `backends`, `tests`

**Classes:**
- PascalCase: `Transcriber`, `ParakeetBackend`, `TranscriptionResult`
- Suffixes indicate role: `*Backend` for implementations, `*Result` for data classes

**Functions:**
- snake_case: `create_backend()`, `resolve_model()`, `format_srt()`
- Private functions prefixed with `_`: `_load_model()`, `_validate_format_capabilities()`

**Constants:**
- UPPER_SNAKE_CASE: `DEFAULT_MODEL`, `SUPPORTED_EXTENSIONS`, `MODEL_REGISTRY`

## Where to Add New Code

**New Backend:**
1. Create `pedantic_parakeet/backends/<name>.py`
2. Implement `BaseTranscriber` protocol
3. Add to `MODEL_REGISTRY` in `pedantic_parakeet/backends/registry.py`
4. Add lazy import in `pedantic_parakeet/backends/__init__.py` (if optional dep)
5. Add case to `create_backend()` in `pedantic_parakeet/transcriber.py`
6. Add tests in `tests/test_backends.py`

**New Output Format:**
1. Add formatter function in `pedantic_parakeet/formatters.py`
2. Add to `FORMATTERS` dict and `EXTENSIONS` dict
3. Format auto-available in CLI

**New CLI Option:**
1. Add parameter to `main()` in `pedantic_parakeet/cli.py`
2. Add validation if needed
3. Pass through to `Transcriber` constructor
4. Add tests in `tests/test_cli.py`

**New Model to Registry:**
1. Add `ModelInfo` entry to `MODEL_REGISTRY` in `pedantic_parakeet/backends/registry.py`
2. Include: model_id, backend, capabilities, description, aliases
3. Add tests in `tests/test_backends.py`

**Utilities:**
- Audio utilities: `pedantic_parakeet/audio.py`
- Shared helpers: Add to appropriate existing module or create new `pedantic_parakeet/<name>.py`

## Special Directories

**`pedantic_parakeet/parakeet_mlx/`:**
- Purpose: Vendored third-party library (parakeet-mlx)
- Generated: No (forked and modified manually)
- Committed: Yes
- Note: Contains `VENDORED.txt` and `LICENSE` documenting origin

**`.venv/`:**
- Purpose: Python virtual environment
- Generated: Yes (via `uv venv`)
- Committed: No (in .gitignore)

**`__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes (by Python)
- Committed: No (in .gitignore)

**`.planning/`:**
- Purpose: GSD planning documents
- Generated: No (manually created)
- Committed: Yes
- Note: Contains project planning, milestones, and codebase analysis

---

*Structure analysis: 2026-01-19*
