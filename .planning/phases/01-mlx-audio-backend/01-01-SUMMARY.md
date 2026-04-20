---
phase: 01-mlx-audio-backend
plan: 01
subsystem: api
tags: [mlx, backend, registry, dataclass, protocol, enum]

# Dependency graph
requires: []
provides:
  - Backend enum with parakeet and mlx-audio values
  - STTCapabilities dataclass for feature flags
  - ModelInfo dataclass with model metadata
  - BaseTranscriber protocol for backend implementations
  - Curated MODEL_REGISTRY with 4 STT models
  - Model resolution with alias support
affects: [01-02, 01-03, cli, transcriber]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Protocol-based backend interface"
    - "Frozen dataclass for immutable capabilities"
    - "Enum with string values for CLI integration"
    - "Registry pattern for model metadata"

key-files:
  created:
    - pedantic_parakeet/backends/__init__.py
    - pedantic_parakeet/backends/base.py
    - pedantic_parakeet/backends/registry.py
  modified: []

key-decisions:
  - "Backend enum uses string values for CLI compatibility"
  - "STTCapabilities is frozen dataclass for immutability"
  - "BaseTranscriber uses Protocol for duck-typing"
  - "Model aliases enable short CLI commands"

patterns-established:
  - "Registry pattern: MODEL_REGISTRY dict with resolve helpers"
  - "Protocol-based interface: BaseTranscriber for backend implementations"

# Metrics
duration: 2min
completed: 2026-01-20
---

# Phase 01 Plan 01: Backend Capability Contracts Summary

**Backend enum, STTCapabilities dataclass, ModelInfo metadata, BaseTranscriber protocol, and curated model registry with 4 STT models**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-20T02:28:49Z
- **Completed:** 2026-01-20T02:30:12Z
- **Tasks:** 2
- **Files created:** 3

## Accomplishments

- Defined Backend enum with `parakeet` and `mlx-audio` values for backend selection
- Created STTCapabilities frozen dataclass with feature flags (timestamps, language_bias, language_hint, chunking)
- Created ModelInfo dataclass with full model metadata including aliases for CLI convenience
- Defined BaseTranscriber Protocol for duck-typed backend implementations
- Built curated MODEL_REGISTRY with 4 STT models: Parakeet v3, Parakeet v2, Whisper Turbo, Voxtral Mini
- Implemented resolver helpers with alias support and helpful error messages

## Task Commits

Each task was committed atomically:

1. **Task 1: Define backend capability contracts** - `cdd7a41` (feat)
2. **Task 2: Build curated model registry and resolver helpers** - `f56e326` (feat)

## Files Created/Modified

- `pedantic_parakeet/backends/__init__.py` - Public exports for backend module
- `pedantic_parakeet/backends/base.py` - Backend enum, STTCapabilities, ModelInfo, BaseTranscriber Protocol
- `pedantic_parakeet/backends/registry.py` - MODEL_REGISTRY, get_model_info(), list_models(), resolve_model()

## Decisions Made

- **Backend enum string values:** Used `str, Enum` base classes so enum values serialize naturally in CLI output
- **Frozen STTCapabilities:** Made immutable to prevent accidental modification of capability flags
- **Protocol for BaseTranscriber:** Allows duck-typing rather than forcing inheritance, more Pythonic
- **Model aliases:** Added short aliases (e.g., "whisper", "parakeet") for CLI convenience while keeping full HF IDs canonical

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Backend contracts and registry ready for CLI integration (Plan 02)
- Model resolution with aliases enables `--model parakeet` shorthand
- Capabilities metadata ready for feature-gating in transcriber

---
*Phase: 01-mlx-audio-backend*
*Completed: 2026-01-20*
