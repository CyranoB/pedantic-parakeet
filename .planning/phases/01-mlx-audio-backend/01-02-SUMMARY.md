---
phase: 01-mlx-audio-backend
plan: 02
subsystem: api
tags: [mlx, backend, parakeet, mlx-audio, factory, optional-dependency]

# Dependency graph
requires:
  - phase: 01-01
    provides: Backend enum, STTCapabilities, ModelInfo, BaseTranscriber protocol, MODEL_REGISTRY
provides:
  - ParakeetBackend implementation with language bias support
  - MlxAudioBackend implementation with dependency guard
  - Transcriber facade with backend factory
  - Optional mlx-audio dependency group
affects: [01-03, cli, transcriber]

# Tech tracking
tech-stack:
  added:
    - mlx-audio>=0.2.5 (optional)
  patterns:
    - "Lazy import pattern for optional dependencies"
    - "Factory pattern for backend instantiation"
    - "Facade pattern for Transcriber class"

key-files:
  created:
    - pedantic_parakeet/backends/parakeet.py
    - pedantic_parakeet/backends/mlx_audio.py
    - pedantic_parakeet/types.py
  modified:
    - pedantic_parakeet/transcriber.py
    - pedantic_parakeet/backends/__init__.py
    - pedantic_parakeet/backends/base.py
    - pyproject.toml

key-decisions:
  - "Extract Token, Segment, TranscriptionResult to types.py to avoid circular imports"
  - "Use lazy imports for mlx-audio to prevent import errors when not installed"
  - "Raise RuntimeError with install guidance when mlx-audio missing"
  - "Handle multiple mlx-audio output formats (Whisper segments, Parakeet sentences)"

patterns-established:
  - "Factory pattern: create_backend() instantiates appropriate backend"
  - "Lazy import pattern: __getattr__ for optional MlxAudioBackend export"
  - "Facade pattern: Transcriber delegates to backend implementations"

# Metrics
duration: 4min
completed: 2026-01-20
---

# Phase 01 Plan 02: Backend Adapters Summary

**ParakeetBackend and MlxAudioBackend implementations with factory wiring and optional dependency support**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-20T02:31:20Z
- **Completed:** 2026-01-20T02:35:29Z
- **Tasks:** 2
- **Files created:** 3
- **Files modified:** 4

## Accomplishments

- Extracted ParakeetBackend class implementing BaseTranscriber protocol with language bias support
- Created MlxAudioBackend with lazy import and clear install guidance when missing
- Built create_backend() factory that routes to appropriate backend based on model registry
- Extracted types (Token, Segment, TranscriptionResult) to separate module to resolve circular imports
- Updated Transcriber facade to delegate to backend implementations
- Added mlx-audio optional dependency group in pyproject.toml

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract Parakeet backend and add Transcriber factory** - `90473d9` (feat)
2. **Task 2: Implement mlx-audio backend + optional dependency** - `70e0534` (feat)

## Files Created/Modified

- `pedantic_parakeet/backends/parakeet.py` - ParakeetBackend implementation wrapping vendored parakeet-mlx
- `pedantic_parakeet/backends/mlx_audio.py` - MlxAudioBackend with lazy import and multi-format output handling
- `pedantic_parakeet/types.py` - Token, Segment, TranscriptionResult dataclasses
- `pedantic_parakeet/transcriber.py` - Transcriber facade with create_backend() factory
- `pedantic_parakeet/backends/__init__.py` - Updated exports with lazy MlxAudioBackend import
- `pedantic_parakeet/backends/base.py` - Updated TYPE_CHECKING import path
- `pyproject.toml` - Added mlx-audio and all optional dependency groups

## Decisions Made

- **Circular import resolution:** Extracted Token, Segment, TranscriptionResult to types.py module since both transcriber.py and backend implementations need these types
- **Lazy import pattern:** MlxAudioBackend checks for mlx-audio availability at instantiation time, not import time, to allow module to import cleanly
- **Error messaging:** RuntimeError with explicit install command: `pip install pedantic-parakeet[mlx-audio]`
- **Multi-format output:** MlxAudioBackend handles various mlx-audio output formats (Whisper segments with words, Parakeet sentences with tokens)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Resolved circular import between transcriber.py and backends**
- **Found during:** Task 1 (ParakeetBackend creation)
- **Issue:** parakeet.py imported from transcriber.py, but transcriber.py now imports from backends, causing circular import
- **Fix:** Created types.py module for shared dataclasses (Token, Segment, TranscriptionResult)
- **Files created:** pedantic_parakeet/types.py
- **Verification:** Import succeeds without errors
- **Committed in:** 90473d9 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (blocking)
**Impact on plan:** Necessary structural change to support backend delegation pattern. No scope creep.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Backend adapters ready for CLI integration (Plan 03)
- Factory pattern enables `--model` and `--backend` CLI options
- MlxAudioBackend ready to use when mlx-audio is installed
- Transcriber facade maintains backwards compatibility

---
*Phase: 01-mlx-audio-backend*
*Completed: 2026-01-20*
