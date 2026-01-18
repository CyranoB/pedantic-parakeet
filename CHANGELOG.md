# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Language bias feature to reduce code-switching for non-English transcription
  - `--language` flag to specify target language (currently supports `fr`)
  - `--language-strength` parameter (0.0-2.0) to control suppression intensity
  - Suppresses common English filler words when transcribing French audio
- Comprehensive test suite with 101 tests covering all major modules
  - `tests/test_alignment.py` - 30 tests for alignment functions
  - `tests/test_cli.py` - 31 tests for CLI subtitle formatting
  - `tests/test_parakeet.py` - 30 tests for decode functions
  - `tests/test_language_bias.py` - 10 tests for language bias feature

### Changed
- Refactored `decode_beam` in `parakeet.py` to reduce cognitive complexity from 37 to ~15
  - Extracted `_create_beam_token()`, `_expand_hypothesis()`, `_process_beam_hypothesis()`
- Refactored alignment functions in `alignment.py`
  - Extracted 8 helper functions from `merge_longest_contiguous` and `merge_longest_common_subsequence`
- Refactored CLI formatting in `cli.py`
  - Extracted 10+ helper functions for subtitle generation and transcription workflow
- Refactored `BaseParakeet` and `ParakeetTDT` classes in `parakeet.py`
  - Extracted 15+ helper methods for decode operations

### Fixed
- 38+ SonarQube code quality issues resolved:
  - **alignment.py**: Redundant `str()` calls, cognitive complexity (35→15, 30→15)
  - **attention.py**: Variable naming conventions (`Tq`→`t_q`, `S_q`→`s_q`, etc.), unused variables
  - **audio.py**: Unused parameters, nested conditionals, shadowed builtins (`bytes`→`audio_bytes`)
  - **cli.py**: Cognitive complexity (23→15, 43→15), unused return values, lambda capture issues
  - **conformer.py**: Commented-out code, nested ternary expressions, TODO comments
  - **parakeet.py**: Cognitive complexity, unused variables, field naming conflicts
  - **test files**: Floating-point equality checks (use `pytest.approx()`), commented code

### Quality
- SonarQube Quality Gate: **Passing**
  - Reliability: A
  - Security: A
  - Maintainability: A
- All 101 tests passing
- Verified transcription output unchanged after refactoring (tested with multiple audio files)

### Known Limitations
- `cli.py` `transcribe` function has 21 parameters (SonarQube limit is 13)
  - Accepted as limitation: CLI commands typically require many user-facing options
