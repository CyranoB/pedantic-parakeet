# Project Milestones: Pedantic Parakeet

## v1.0 MVP (Shipped: 2026-01-20)

**Delivered:** Multi-backend transcription support with curated model registry and capability-based validation.

**Phases completed:** 1 (3 plans total)

**Key accomplishments:**

- Multi-backend architecture with Backend enum, STTCapabilities contracts, and curated model registry
- ParakeetBackend (with language bias) and MlxAudioBackend (lazy loading, dependency guard)
- CLI backend selection with `--backend`, `--model`, `--list-models` flags and alias support
- Capability-based validation (format validation for text-only models, language validation)
- Comprehensive test coverage (145 tests covering registry, CLI, backends, and language bias)

**Stats:**

- 10 files created/modified
- +1,175 lines of Python
- 1 phase, 3 plans
- 3 days from start to ship

**Git range:** `cdd7a41` → `4a215ce`

**What's next:** TBD — run `/gsd-new-milestone` to plan v1.1

---

*For full milestone details, see `.planning/milestones/v1.0-ROADMAP.md`*
