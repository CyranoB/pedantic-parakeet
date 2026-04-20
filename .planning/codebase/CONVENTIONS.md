# Coding Conventions

**Analysis Date:** 2026-01-19

## Naming Patterns

**Files:**
- Snake_case for all Python files: `language_bias.py`, `test_backends.py`
- Module directories use snake_case: `parakeet_mlx/`, `backends/`
- Test files prefix with `test_`: `test_cli.py`, `test_alignment.py`

**Functions:**
- Snake_case for all functions: `build_language_bias()`, `tokens_to_sentences()`
- Private functions prefix with underscore: `_format_timestamp_srt()`, `_load_model()`
- Factory functions use `create_` or `from_` prefix: `create_backend()`, `from_pretrained()`

**Variables:**
- Snake_case for local variables and module constants
- UPPER_SNAKE_CASE for module-level constants: `SUPPORTED_LANGUAGES`, `MODEL_REGISTRY`
- Single underscore prefix for private attributes: `self._model`, `self._capabilities`

**Types:**
- PascalCase for dataclasses and classes: `AlignedToken`, `TranscriptionResult`, `Transcriber`
- PascalCase for Enums: `Backend`
- Protocol classes use `Base` prefix: `BaseTranscriber`

**Arguments:**
- Snake_case for function parameters
- Optional parameters use `| None` type hint: `language: str | None = None`

## Code Style

**Formatting:**
- Ruff formatter (configured in `pyproject.toml`)
- Line length: 100 characters
- Target Python version: 3.10

**Linting:**
- Ruff linter with rules: E (errors), F (pyflakes), I (isort), W (warnings)
- Configuration in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

## Import Organization

**Order:**
1. Standard library imports (`from pathlib import Path`, `import json`)
2. Third-party imports (`import typer`, `from rich.console import Console`)
3. Local imports (`from .backends.base import Backend`, `from ..types import Token`)

**Path Aliases:**
- No path aliases used
- Relative imports within package: `from .backends.base import Backend`
- Absolute imports for external packages: `import mlx.core as mx`

**Import Style:**
```python
# Standard library
from collections.abc import Callable
from pathlib import Path

# Third-party
import typer
from rich.console import Console

# Local - relative imports within package
from .backends.base import Backend, STTCapabilities
from ..types import Segment, Token, TranscriptionResult
```

## Type Annotations

**Pattern:**
- All public functions have type annotations
- Use `list[X]` not `List[X]` (modern Python 3.10+ style)
- Use `X | None` not `Optional[X]`
- Use `Annotated` for CLI options with Typer

**Examples:**
```python
def transcribe(
    self,
    audio_path: Path | str,
    chunk_callback: Callable[[float, float], None] | None = None,
) -> TranscriptionResult:
    ...

def list_models(backend: Backend | None = None) -> list[ModelInfo]:
    ...
```

**TYPE_CHECKING Pattern:**
- Use `TYPE_CHECKING` guard for imports only needed for type hints:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from ..types import TranscriptionResult
```

## Error Handling

**Patterns:**
- Raise `ValueError` for invalid arguments with descriptive messages
- Raise `RuntimeError` for runtime failures (e.g., missing dependencies)
- Use `typer.BadParameter` for CLI validation errors
- Include helpful context in error messages:
```python
raise ValueError(
    f"Unknown model: '{model_id}'. "
    f"Supported models: {supported}. "
    f"Aliases: {aliases}."
)
```

**Validation:**
- Validate early in function entry
- CLI validates before instantiating backends
- Use assertions sparingly (only for internal invariants)

## Logging

**Framework:** warnings module for user warnings
- No logging framework configured
- Use `warnings.warn()` for non-fatal issues:
```python
if suppressed == 0:
    warnings.warn("No English tokens found to suppress")
```

**Console Output:**
- Use Rich console for CLI output: `Console()`, `Console(stderr=True)`
- Colored output with Rich markup: `[red]Error[/red]`, `[green]Success[/green]`

## Comments

**When to Comment:**
- Module docstrings explain purpose and provide context
- Complex algorithms get explanatory comments
- TODO comments for known improvements (rare in codebase)

**Docstring Style:**
- Google-style docstrings for public functions
- Args/Returns/Raises sections:
```python
def resolve_model(model_id: str) -> ModelInfo:
    """Resolve a model ID or alias to ModelInfo.

    Args:
        model_id: Full model ID or short alias.

    Returns:
        ModelInfo for the resolved model.

    Raises:
        ValueError: If model ID is not found in registry.
    """
```

**Attribute Documentation:**
- Use docstrings for dataclass attributes:
```python
@dataclass
class ModelInfo:
    model_id: str
    """HuggingFace model identifier."""

    backend: Backend
    """Which transcription backend to use."""
```

## Function Design

**Size:**
- Functions kept reasonably small (most under 30 lines)
- Extract helper functions for complex logic: `_get_overlap_regions()`, `_merge_by_cutoff()`
- Single responsibility per function

**Parameters:**
- Keyword-only for optional parameters when clarity needed: `*, overlap_duration: float`
- Default values for optional parameters
- Group related parameters in dataclasses: `SentenceConfig`, `DecodingConfig`

**Return Values:**
- Single return type per function
- Use dataclasses for structured returns: `TranscriptionResult`, `AlignedResult`
- Properties for computed values: `@property def duration(self) -> float:`

## Module Design

**Exports:**
- Use `__all__` for explicit public API:
```python
__all__ = [
    "Token",
    "Segment",
    "TranscriptionResult",
    "Transcriber",
    "create_backend",
    "DEFAULT_MODEL",
]
```

**Barrel Files:**
- Package `__init__.py` re-exports public API
- Example: `pedantic_parakeet/__init__.py` exports main classes

**Organization:**
- Related functionality in single module
- Dataclasses near the code that uses them
- Separate `types.py` for shared data structures

## Dataclass Patterns

**Standard Pattern:**
```python
@dataclass
class Token:
    """A word or subword token with timing information."""
    text: str
    start: float  # seconds
    end: float  # seconds
    confidence: float = 1.0
```

**Frozen Dataclasses:**
- Use `frozen=True` for immutable configuration:
```python
@dataclass(frozen=True)
class STTCapabilities:
    supports_timestamps: bool = True
    supports_language_bias: bool = False
```

**Post-init Processing:**
- Use `__post_init__` for computed fields:
```python
def __post_init__(self) -> None:
    self.end = self.start + self.duration
```

**Factory Defaults:**
- Use `field(default_factory=list)` for mutable defaults:
```python
aliases: list[str] = field(default_factory=list)
tokens: list[Token] = field(default_factory=list)
```

## Protocol Pattern

**Runtime Checkable Protocols:**
```python
@runtime_checkable
class BaseTranscriber(Protocol):
    """Protocol defining the transcriber interface."""

    @property
    def model_id(self) -> str:
        ...

    def transcribe(
        self,
        audio_path: "Path | str",
        chunk_callback: "Callable[[float, float], None] | None" = None,
    ) -> "TranscriptionResult":
        ...
```

## Lazy Loading Pattern

**Model Loading:**
- Defer expensive operations to first use:
```python
def _load_model(self) -> BaseParakeet:
    """Lazy load the model on first use."""
    if self._model is None:
        self._model = from_pretrained(self._model_id)
    return self._model
```

**Backend Creation:**
```python
def _get_backend(self) -> BaseTranscriber:
    """Lazy create the backend on first use."""
    if self._backend is None:
        self._backend = create_backend(**self._backend_params)
    return self._backend
```

## Enum Pattern

**String Enum:**
```python
class Backend(str, Enum):
    """Supported transcription backends."""

    PARAKEET = "parakeet"
    MLX_AUDIO = "mlx-audio"

    def __str__(self) -> str:
        return self.value
```

## Constants Pattern

**Module-level Constants:**
```python
# Formats supported by librosa/ffmpeg
SUPPORTED_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".wma"
})

# Schema version for JSON output
JSON_SCHEMA_VERSION = "1.0"

# Default model
DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
```

**Registry Pattern:**
```python
MODEL_REGISTRY: dict[str, ModelInfo] = {
    "mlx-community/parakeet-tdt-0.6b-v3": ModelInfo(
        model_id="mlx-community/parakeet-tdt-0.6b-v3",
        backend=Backend.PARAKEET,
        capabilities=STTCapabilities(...),
        description="...",
        aliases=["parakeet-v3", "parakeet"],
    ),
    # ... more entries
}
```

---

*Convention analysis: 2026-01-19*
