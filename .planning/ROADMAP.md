# Roadmap: Pedantic Parakeet

## Overview

Pedantic Parakeet delivers high-quality audio transcription for Apple Silicon. Phase 1 established multi-backend support with a curated model registry. The v1.0 MVP requirements are complete — the tool is production-ready for transcription workflows.

## Milestones

- ✅ **v1.0 MVP** - Phase 1 (shipped 2026-01-20)

## Phases

### Phase 1: mlx-audio Backend Support ✓
**Goal**: Enable transcription using multiple MLX backends with capability-based validation
**Depends on**: Nothing (first phase)
**Requirements**: BACK-01, BACK-02, BACK-03, BACK-04, BACK-05, OUT-06, LANG-03, CLI-05, QA-01, QA-02
**Success Criteria** (what must be TRUE):
  1. User can list curated models showing backend and capability info
  2. User can select backend with `--backend` and models with `--model` or aliases
  3. Text-only models (Voxtral) reject srt/vtt/json formats with clear error
  4. Language bias options are validated against model capabilities
  5. Backend registry and CLI validation have test coverage
**Plans**: 3 plans

Plans:
- [x] 01-01: Backend capability contracts and curated model registry
- [x] 01-02: Parakeet and mlx-audio backend adapters with factory wiring
- [x] 01-03: CLI flags, validation, tests, and documentation

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. mlx-audio Backend | 3/3 | Complete | 2026-01-20 |

---

## Future Considerations (v1.1+)

Potential future work (not planned):

- **Additional language bias support** — expand beyond French
- **Speaker diarization** — identify different speakers
- **Real-time transcription** — streaming audio input
- **Model fine-tuning support** — custom vocabularies
- **GUI application** — desktop interface

---

*Created: 2026-01-20*
*Last updated: 2026-01-20*
