---
phase: 01-mlx-audio-backend
plan: 03
subsystem: cli
tags: [cli, typer, validation, testing, pytest]

# Dependency graph
requires:
  - phase: 01-01
    provides: Backend enum, STTCapabilities, ModelInfo, MODEL_REGISTRY
  - phase: 01-02
    provides: ParakeetBackend, MlxAudioBackend, Transcriber facade
provides:
  - CLI --backend and --list-models flags
  - Format validation for timestamp-less models
  - Language option validation per model capabilities
  - Test coverage for registry and CLI validation
  - README documentation for backend selection
affects: [cli-users, documentation]

# Tech tracking
tech-stack:
  added:
    - typer.testing.CliRunner (for CLI tests)
  patterns:
    - "Early validation pattern: validate before backend instantiation"
    - "Eager callback pattern: --list-models exits before argument parsing"

key-files:
  created:
    - tests/test_backends.py
  modified:
    - pedantic_parakeet/cli.py
    - tests/test_cli.py
    - README.md

key-decisions:
  - "Validation before instantiation to avoid importing mlx-audio unnecessarily"
  - "Unknown models pass validation (let Transcriber handle unknown model errors)"
  - "Use is_eager callback for --list-models to exit before requiring inputs argument"

patterns-established:
  - "Early validation pattern: _validate_format_capabilities and _validate_language_capabilities run before Transcriber creation"
  - "Eager callback pattern: --list-models uses is_eager=True to exit cleanly"

# Metrics
duration: 4min
completed: 2026-01-20
---

# Phase 01 Plan 03: CLI Integration & Test Coverage Summary

**CLI backend selection with --backend and --list-models flags, format/language validation, and comprehensive test coverage**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-20T02:36:51Z
- **Completed:** 2026-01-20T02:40:35Z
- **Tasks:** 2
- **Files created:** 1
- **Files modified:** 3

## Accomplishments

- Added `--backend` option with choices `parakeet` and `mlx-audio` (auto-detected by default)
- Added `--list-models` flag that prints curated models with backend, timestamp support, and aliases
- Implemented format validation that rejects srt/vtt/json for models without timestamps (e.g., Voxtral)
- Implemented language validation that rejects `--language-strength` for models without language bias support
- Created comprehensive test suite for backend registry (28 tests) and CLI validation (19 new tests)
- Updated README with mlx-audio optional install instructions and backend selection documentation

## Task Commits

Each task was committed atomically:

1. **Task 1: Add CLI backend flags, validation, and docs** - `3ed8279` (feat)
2. **Task 2: Add registry + CLI validation tests** - `e9a9305` (test)

## Files Created/Modified

- `tests/test_backends.py` - New test file for registry resolution, backend assignment, and capability validation
- `pedantic_parakeet/cli.py` - Added --backend, --list-models, and validation functions
- `tests/test_cli.py` - Extended with CLI validation tests (--list-models output, format/language validation)
- `README.md` - Added mlx-audio optional install and backend selection documentation

## Decisions Made

- **Validation before instantiation:** Format and language validations run before creating Transcriber to avoid importing mlx-audio when not installed
- **Unknown models pass validation:** Validation functions allow unknown model IDs through (Transcriber will provide proper error handling for unknown models)
- **Eager callback for --list-models:** Uses `is_eager=True` to exit cleanly before the `inputs` argument is required

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 01 complete: Backend capability contracts, adapters, and CLI integration all implemented
- Users can now:
  - List available models with `--list-models`
  - Select backend with `--backend parakeet` or `--backend mlx-audio`
  - Use aliases like `--model whisper` or `--model voxtral`
- Validation ensures safe usage:
  - Text-only models (Voxtral) can only output `txt` format
  - Language bias options only work with Parakeet backend
- Ready for user testing and feedback

---
*Phase: 01-mlx-audio-backend*
*Completed: 2026-01-20*
