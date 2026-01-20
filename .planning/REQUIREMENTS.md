# Requirements: Pedantic Parakeet v1.0

## Milestone: v1.0 MVP

### Backend Support (BACK)

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| BACK-01 | User can transcribe audio using vendored Parakeet backend (default) | Must | Complete |
| BACK-02 | User can transcribe audio using mlx-audio backend (Whisper, Voxtral) | Must | Complete |
| BACK-03 | User can list available models with `--list-models` | Must | Complete |
| BACK-04 | User can select backend explicitly with `--backend` flag | Must | Complete |
| BACK-05 | User can use model aliases for convenience (e.g., `whisper` instead of full HF ID) | Should | Complete |

### Output Formats (OUT)

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| OUT-01 | User can output plain text transcripts (txt) | Must | Complete |
| OUT-02 | User can output SubRip subtitles (srt) | Must | Complete |
| OUT-03 | User can output WebVTT subtitles (vtt) | Must | Complete |
| OUT-04 | User can output structured JSON with timestamps | Must | Complete |
| OUT-05 | User can output multiple formats in a single run | Should | Complete |
| OUT-06 | Text-only models (Voxtral) are validated to reject srt/vtt/json | Must | Complete |

### Language Bias (LANG)

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| LANG-01 | User can bias transcription toward target language (French) | Should | Complete |
| LANG-02 | User can adjust language bias strength (0.0-2.0) | Should | Complete |
| LANG-03 | Language bias validation rejects unsupported backends | Must | Complete |

### CLI & UX (CLI)

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| CLI-01 | User can process single audio files | Must | Complete |
| CLI-02 | User can batch process directories of audio files | Must | Complete |
| CLI-03 | User can preview processing with `--dry-run` | Should | Complete |
| CLI-04 | User sees progress bars for long transcriptions | Should | Complete |
| CLI-05 | User receives clear error messages for invalid options | Must | Complete |
| CLI-06 | User can specify output directory with `--output` | Should | Complete |

### Quality & Testing (QA)

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| QA-01 | Backend registry has test coverage | Must | Complete |
| QA-02 | CLI validation has test coverage | Must | Complete |
| QA-03 | Language bias has test coverage | Must | Complete |

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BACK-01 | Phase 1 | Complete |
| BACK-02 | Phase 1 | Complete |
| BACK-03 | Phase 1 | Complete |
| BACK-04 | Phase 1 | Complete |
| BACK-05 | Phase 1 | Complete |
| OUT-01 | Pre-v1 | Complete |
| OUT-02 | Pre-v1 | Complete |
| OUT-03 | Pre-v1 | Complete |
| OUT-04 | Pre-v1 | Complete |
| OUT-05 | Pre-v1 | Complete |
| OUT-06 | Phase 1 | Complete |
| LANG-01 | Pre-v1 | Complete |
| LANG-02 | Pre-v1 | Complete |
| LANG-03 | Phase 1 | Complete |
| CLI-01 | Pre-v1 | Complete |
| CLI-02 | Pre-v1 | Complete |
| CLI-03 | Pre-v1 | Complete |
| CLI-04 | Pre-v1 | Complete |
| CLI-05 | Phase 1 | Complete |
| CLI-06 | Pre-v1 | Complete |
| QA-01 | Phase 1 | Complete |
| QA-02 | Phase 1 | Complete |
| QA-03 | Pre-v1 | Complete |

---

*Requirements derived from README.md, PLAN.md, and codebase analysis*
*Created: 2026-01-20*
