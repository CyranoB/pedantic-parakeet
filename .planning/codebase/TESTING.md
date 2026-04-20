# Testing Patterns

**Analysis Date:** 2026-01-19

## Test Framework

**Runner:**
- pytest (installed as dev dependency)
- Config in `pyproject.toml` (minimal, just enables pytest)
- No pytest.ini or conftest.py files found

**Assertion Library:**
- pytest built-in assertions
- `pytest.approx()` for floating point comparisons

**Run Commands:**
```bash
pytest                    # Run all tests
pytest tests/test_cli.py  # Run specific test file
pytest -v                 # Verbose output
pytest -x                 # Stop on first failure
```

## Test File Organization

**Location:**
- Separate `tests/` directory at project root
- Tests not co-located with source code

**Naming:**
- Files: `test_<module>.py`
- Classes: `Test<Feature>` (e.g., `TestFormatTimestamp`, `TestModelRegistry`)
- Methods: `test_<description>` (e.g., `test_zero_seconds`, `test_returns_valid_json`)

**Structure:**
```
tests/
├── __init__.py
├── test_alignment.py      # Tests for alignment module
├── test_backends.py       # Tests for backend registry
├── test_cli.py            # Tests for CLI and formatters
├── test_language_bias.py  # Tests for language bias
└── test_parakeet.py       # Tests for parakeet model functions
```

## Test Structure

**Suite Organization:**
```python
"""Tests for cli.py formatting functions."""

import pytest
from typer.testing import CliRunner

from pedantic_parakeet.parakeet_mlx.alignment import AlignedResult

runner = CliRunner()


# Helper to create test data
def make_token(id: int, text: str, start: float, duration: float, confidence: float = 1.0) -> AlignedToken:
    return AlignedToken(id=id, text=text, start=start, duration=duration, confidence=confidence)


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_zero_seconds(self):
        result = format_timestamp(0.0)
        assert result == "00:00:00,000"

    def test_simple_seconds(self):
        result = format_timestamp(5.0)
        assert result == "00:00:05,000"
```

**Patterns:**
- Module docstring describing test scope
- Helper functions at module level for creating test data
- Class-based organization grouping related tests
- Each test method tests one specific behavior
- Descriptive test names that explain expected behavior

## Fixtures

**pytest Fixtures:**
```python
@pytest.fixture
def mock_base_parakeet(self):
    """Create a mock BaseParakeet with minimal setup."""
    with patch.object(BaseParakeet, '__init__', lambda self, *args, **kwargs: None):
        model = BaseParakeet.__new__(BaseParakeet)
        # Set up minimal required attributes
        model.preprocessor_config = MagicMock()
        model.preprocessor_config.sample_rate = 16000
        model.preprocessor_config.hop_length = 160
        model.encoder_config = MagicMock()
        model.encoder_config.subsampling_factor = 4
        return model

@pytest.fixture
def mock_model_for_transcribe(self):
    """Create mock model for transcribe tests."""
    model = MagicMock()
    model.preprocessor_config = MagicMock()
    model.preprocessor_config.sample_rate = 16000
    model.generate = MagicMock(return_value=[
        AlignedResult(text="Hello world", sentences=[])
    ])
    return model
```

**Factory Functions:**
- Helper functions create test data consistently:
```python
def make_token(id: int, text: str, start: float, duration: float, confidence: float = 1.0) -> AlignedToken:
    return AlignedToken(id=id, text=text, start=start, duration=duration, confidence=confidence)


def make_simple_result() -> AlignedResult:
    """Create a simple result with one sentence."""
    tokens = [
        make_token(1, "Hello", 0.0, 0.5, confidence=0.95),
        make_token(2, " world", 0.5, 0.5, confidence=0.90),
    ]
    sentence = AlignedSentence(text="Hello world", tokens=tokens)
    return AlignedResult(text="Hello world", sentences=[sentence])
```

## Mocking

**Framework:** unittest.mock (via `from unittest.mock import MagicMock, patch`)

**Patterns:**
```python
from unittest.mock import MagicMock, patch

# Mock an object's __init__
with patch.object(BaseParakeet, '__init__', lambda self, *args, **kwargs: None):
    model = BaseParakeet.__new__(BaseParakeet)

# Create mock with attributes
model = MagicMock()
model.vocabulary = ["a", "b", "c", "d"]
model.durations = [0, 1, 2, 4]
model.time_ratio = 0.04

# Mock method calls
mock_tdt_model.decode_greedy = MagicMock(return_value=([], []))
mock_tdt_model.decode_greedy.assert_called_once()
```

**What to Mock:**
- External dependencies (mlx.core arrays)
- Model loading operations
- Complex class initialization
- Methods being tested for routing behavior

**What NOT to Mock:**
- Dataclasses (use real instances)
- Simple helper functions
- The function under test

## Assertion Patterns

**Basic Assertions:**
```python
assert result == "00:00:00,000"
assert len(sentences) == 2
assert "Hello world" in txt
```

**Approximate Comparisons:**
```python
assert token.start == pytest.approx(0.5)
assert confidence > 0.9
assert 0.84 < sentence.confidence < 0.86
```

**Exception Testing:**
```python
with pytest.raises(ValueError, match="not supported"):
    build_language_bias(["a"], "xx", 0.5)

with pytest.raises(RuntimeError):
    merge_longest_contiguous(a, b, overlap_duration=1.0)

with pytest.raises(typer.BadParameter) as exc_info:
    _validate_format_capabilities(["srt"], "voxtral")
error_msg = str(exc_info.value)
assert "does not support timestamps" in error_msg
```

**Warning Testing:**
```python
with pytest.warns(UserWarning, match="No English tokens found"):
    build_language_bias(vocab, "fr", 0.5)
```

## Test Categories

**Unit Tests:**
- Test individual functions in isolation
- Most tests in codebase are unit tests
- Example: `test_format_timestamp.py` tests timestamp formatting functions

**Integration Tests:**
- Test multiple components working together
- Example: `TestIntegration` class in `test_alignment.py`:
```python
class TestIntegration:
    """Integration tests for the alignment module."""

    def test_full_pipeline(self):
        """Test complete flow: tokens -> sentences -> result."""
        tokens = [
            make_token(1, "Hello", 0.0, 0.5, confidence=0.95),
            # ...
        ]
        sentences = tokens_to_sentences(tokens)
        result = sentences_to_result(sentences)
        
        assert len(result.sentences) == 2
        assert "Hello world" in result.text
```

**CLI Tests:**
- Uses `typer.testing.CliRunner`:
```python
runner = CliRunner()

class TestListModels:
    """Tests for --list-models command."""

    def test_list_models_shows_curated_models(self):
        """--list-models should display curated model IDs."""
        result = runner.invoke(app, ["--list-models"])
        assert result.exit_code == 0
        assert "mlx-community/parakeet-tdt-0.6b-v3" in result.output
```

**E2E Tests:**
- Not present in codebase
- Would require actual model loading and audio files

## Test Data

**Inline Test Data:**
```python
def test_english_suppression():
    """English tokens should have negative bias."""
    vocab = ["▁le", "▁the", "▁la"]
    bias = build_language_bias(vocab, "fr", 0.5)
    assert abs(float(bias[1]) - (-0.5)) < 1e-6
```

**Factory Functions:**
```python
def make_simple_result() -> AlignedResult:
    """Create a simple result with one sentence."""
    tokens = [
        make_token(1, "Hello", 0.0, 0.5, confidence=0.95),
        make_token(2, " world", 0.5, 0.5, confidence=0.90),
    ]
    sentence = AlignedSentence(text="Hello world", tokens=tokens)
    return AlignedResult(text="Hello world", sentences=[sentence])
```

**No External Fixtures:**
- No JSON/YAML fixture files
- All test data created in code
- No shared fixture files beyond factory functions

## Coverage

**Requirements:** None enforced

**Current State:**
- No coverage configuration in `pyproject.toml`
- No coverage reports generated by default

**Generate Coverage:**
```bash
pytest --cov=pedantic_parakeet --cov-report=html
```

## Edge Case Testing

**Pattern:** Dedicated test classes for edge cases:
```python
class TestEdgeCases:
    """Edge case tests for formatting functions."""

    def test_empty_sentences(self):
        tokens = [make_token(1, "", 0.0, 0.1)]
        sentence = AlignedSentence(text="", tokens=tokens)
        result = AlignedResult(text="", sentences=[sentence])
        
        # Should not crash
        to_txt(result)
        to_srt(result)

    def test_long_duration(self):
        # 2 hours
        tokens = [make_token(1, "Long", 0.0, 7200.0)]
        # ...

    def test_unicode_text(self):
        tokens = [make_token(1, "Привет мир", 0.0, 1.0)]
        # ...

    def test_special_characters(self):
        tokens = [make_token(1, "Hello <world> & \"friends\"", 0.0, 1.0)]
        # ...
```

## Boundary Testing

**Pattern:** Test boundary values explicitly:
```python
def test_boundary_strength_values():
    """Boundary values (0.0 and 2.0) should be accepted."""
    vocab = ["▁the"]
    # Should not raise
    bias_zero = build_language_bias(vocab, "fr", 0.0)
    bias_max = build_language_bias(vocab, "fr", 2.0)

    assert abs(float(bias_zero[0]) - 0.0) < 1e-6
    assert abs(float(bias_max[0]) - (-2.0)) < 1e-6
```

## Validation Testing

**Pattern:** Test validation error messages contain helpful info:
```python
def test_raises_value_error_for_unknown_model(self):
    """Should raise ValueError with helpful message for unknown model."""
    with pytest.raises(ValueError) as exc_info:
        resolve_model("unknown/model-id")

    error_msg = str(exc_info.value)
    assert "Unknown model" in error_msg
    assert "unknown/model-id" in error_msg
    assert "Supported models" in error_msg
    assert "Aliases" in error_msg
```

## Test Class Organization

**Docstring Pattern:**
```python
class TestFormatTimestamp:
    """Tests for format_timestamp function."""

class TestMergeLongestContiguous:
    """Tests for merge_longest_contiguous function."""

class TestCapabilities:
    """Tests for capability flags in registry."""
```

**Naming Convention:**
- `Test<FunctionName>` for function-specific tests
- `Test<Feature>` for feature-area tests
- `TestEdgeCases`, `TestIntegration` for special categories

## Helper Method Pattern

**Private helpers in test classes:**
```python
class TestDecodeGreedyLogic:
    """Tests for greedy decoding logic."""

    def _add_token_if_non_blank(self, token: int, blank_id: int, hypothesis: list) -> list:
        """Helper to add token if not blank."""
        if token != blank_id:
            hypothesis.append(token)
        return hypothesis

    def test_blank_token_not_added(self):
        vocabulary = ["a", "b", "c"]
        blank_token_id = len(vocabulary)
        hypothesis = []
        hypothesis = self._add_token_if_non_blank(blank_token_id, blank_token_id, hypothesis)
        assert len(hypothesis) == 0
```

## Test Dependencies

**Dependencies (dev):**
- pytest
- typer (for CliRunner)
- unittest.mock (standard library)

**Not Used:**
- pytest-cov (coverage)
- pytest-xdist (parallel)
- hypothesis (property-based)

## Known Issues

**MLX Dependency:**
- Tests require `mlx` module which is Apple Silicon only
- Tests will fail to collect on non-Apple systems
- No mocking strategy for mlx.core at import time

**Import-Time Dependencies:**
- Some test files import modules that require `mlx` at import time
- This causes collection errors before tests can run

---

*Testing analysis: 2026-01-19*
