# Technology Stack

**Analysis Date:** 2026-01-19

## Languages

**Primary:**
- Python >=3.10 - Core application language for all source code

**Secondary:**
- None - Pure Python project

## Runtime

**Environment:**
- Python 3.10+ (verified 3.11 in .venv, project requires >=3.10)
- Apple Silicon optimized via MLX framework

**Package Manager:**
- pip with pyproject.toml (PEP 517/518 compliant)
- Lockfile: Not present (relies on version constraints in pyproject.toml)

**Build System:**
- hatchling - Modern Python build backend

## Frameworks

**Core:**
- MLX >=0.21.0 - Apple Silicon ML framework for neural network inference
- Typer >=0.9.0 - CLI framework with type hints
- Rich >=13.0.0 - Terminal output formatting and progress bars

**Audio Processing:**
- librosa - Audio loading and mel-spectrogram generation
- ffmpeg (external) - Audio format conversion and decoding

**Testing:**
- pytest - Test framework (dev dependency)

**Code Quality:**
- ruff - Linting and formatting (dev dependency)
- Line length: 100 characters
- Target: Python 3.10

## Key Dependencies

**Critical (Core Functionality):**
- `mlx>=0.21.0` - Neural network inference on Apple Silicon
- `numpy` - Array operations and audio data handling
- `huggingface-hub` - Model downloading and caching
- `dacite` - Dataclass instantiation from dictionaries
- `librosa` - Mel-spectrogram computation

**CLI:**
- `typer>=0.9.0` - Command-line interface with type annotations
- `rich>=13.0.0` - Rich text output and progress bars

**Optional:**
- `mlx-audio>=0.2.5` - Alternative STT backend for Whisper models

## Configuration

**Environment:**
- No .env files required
- No runtime secrets
- Models downloaded from HuggingFace Hub (public, no auth needed)
- Model cache location: Default HuggingFace cache (~/.cache/huggingface/)

**Build:**
- `pyproject.toml` - Project metadata, dependencies, and tool configuration
- `[tool.ruff]` - Linting rules in pyproject.toml
- No separate config files for tools

**CLI Entry Point:**
- `pedantic-parakeet` command installed via `[project.scripts]`
- Entry point: `pedantic_parakeet.cli:app`

## Platform Requirements

**Development:**
- Python 3.10+
- ffmpeg installed and in PATH (required for audio loading)
- Apple Silicon Mac (M1/M2/M3) - MLX is Apple Silicon exclusive

**Production:**
- Same as development - CLI tool runs locally
- No server deployment
- No containerization configured

## Optional Dependencies

**Install Commands:**
```bash
# Base install
pip install .

# With dev tools
pip install .[dev]

# With mlx-audio backend
pip install .[mlx-audio]

# All optional dependencies
pip install .[all]
```

## Version Information

- Package version: 0.1.0
- Python: >=3.10
- MLX: >=0.21.0
- License: MIT

---

*Stack analysis: 2026-01-19*
