# Codebase Concerns

**Analysis Date:** 2026-01-19

## Tech Debt

**Vendored Parakeet-MLX Implementation:**
- Issue: The `pedantic_parakeet/parakeet_mlx/` directory contains a vendored copy of the parakeet-mlx library rather than using it as a dependency
- Files: `pedantic_parakeet/parakeet_mlx/*.py` (1198 lines in parakeet.py alone)
- Impact: Updates to upstream parakeet-mlx require manual merging; duplicated maintenance burden
- Fix approach: Consider using parakeet-mlx as a proper dependency if upstream becomes stable, or document sync process

**Hacky Punctuation Detection:**
- Issue: Sentence splitting uses hardcoded character checks with comment "# hacky, will fix"
- Files: `pedantic_parakeet/parakeet_mlx/alignment.py:67-78`
- Impact: May not handle all punctuation correctly across languages; maintenance burden
- Fix approach: Use proper Unicode punctuation detection or sentence boundary detection library

**Dual CLI Entry Points:**
- Issue: Two separate CLI implementations exist - main CLI (`pedantic_parakeet/cli.py`) and vendored CLI (`pedantic_parakeet/parakeet_mlx/cli.py`)
- Files: `pedantic_parakeet/cli.py` (501 lines), `pedantic_parakeet/parakeet_mlx/cli.py` (594 lines)
- Impact: Code duplication; confusing for maintenance; different feature sets
- Fix approach: Consolidate CLI functionality into single entry point

**Empty Greedy Dataclass:**
- Issue: `Greedy` dataclass is empty with just `pass`
- Files: `pedantic_parakeet/parakeet_mlx/parakeet.py:77-78`
- Impact: No configuration options for greedy decoding; inconsistent API with Beam class
- Fix approach: Add configuration options (e.g., temperature) or convert to sentinel value

## Known Bugs

**None identified through static analysis.**
- The codebase appears well-structured with no obvious bugs from code review
- Testing would be needed to identify runtime issues

## Security Considerations

**Subprocess Execution Without Sanitization:**
- Risk: `ffprobe` subprocess call uses user-provided path without validation
- Files: `pedantic_parakeet/audio.py:66-80`
- Current mitigation: Timeout of 10 seconds limits impact
- Recommendations: Validate path is an existing file before passing to subprocess; consider using pathlib's resolve()

**Model Loading from HuggingFace:**
- Risk: Models downloaded from HuggingFace Hub could contain malicious code
- Files: `pedantic_parakeet/parakeet_mlx/parakeet.py` (via `from_pretrained`)
- Current mitigation: None
- Recommendations: Document trusted model sources; consider model hash verification

## Performance Bottlenecks

**Sequential Batch Processing:**
- Problem: TDT and RNNT decoding processes batches sequentially in a for loop
- Files: `pedantic_parakeet/parakeet_mlx/parakeet.py:731-739` (TDT), `pedantic_parakeet/parakeet_mlx/parakeet.py:796-854` (RNNT)
- Cause: Cannot parallelize autoregressive decoding across batch items
- Improvement path: For batch size > 1, consider parallel processing at encoder level; document that batch size 1 is recommended

**Beam Search Memory Growth:**
- Problem: Beam search expands all combinations of top-k tokens and durations
- Files: `pedantic_parakeet/parakeet_mlx/parakeet.py:588-596`
- Cause: Creates `beam_token * beam_duration` candidates per step per hypothesis
- Improvement path: Implement pruning earlier in expansion; consider nucleus sampling

**Large File Complexity:**
- Problem: `parakeet.py` at 1198 lines is difficult to navigate
- Files: `pedantic_parakeet/parakeet_mlx/parakeet.py`
- Cause: All model variants (TDT, RNNT, CTC, Streaming) in single file
- Improvement path: Split into separate files per model type (tdt.py, rnnt.py, ctc.py)

## Fragile Areas

**Streaming Inference Context Manager:**
- Files: `pedantic_parakeet/parakeet_mlx/parakeet.py:1012-1199`
- Why fragile: Complex state management across audio chunks; attention model swapping; cache rotation
- Safe modification: Always test with long audio files; verify memory doesn't grow indefinitely
- Test coverage: No dedicated streaming tests found

**Local Attention Metal Kernels:**
- Files: `pedantic_parakeet/parakeet_mlx/attention.py:233-394`, `pedantic_parakeet/parakeet_mlx/attention.py:395-534`
- Why fragile: Hand-written Metal GPU kernels with manual loop unrolling; hard to debug
- Safe modification: Requires Metal/GPU expertise; test on different sequence lengths
- Test coverage: No unit tests for kernel correctness

**Chunk Overlap Merging:**
- Files: `pedantic_parakeet/parakeet_mlx/alignment.py:116-325`
- Why fragile: Multiple merge strategies (contiguous, LCS); relies on token ID and timing matching
- Safe modification: Extensive edge case testing needed; verify merge behavior with real audio
- Test coverage: `tests/test_alignment.py` covers basic cases but edge cases may exist

## Scaling Limits

**Memory for Long Audio:**
- Current capacity: Chunking with 120s default + 15s overlap
- Limit: Without chunking, very long audio will exhaust GPU memory
- Scaling path: Chunking is already implemented; could add adaptive chunk sizing based on available memory

**Positional Encoding Buffer:**
- Current capacity: `max_len=5000` for positional encodings
- Limit: Sequences longer than 5000 frames (~50 seconds at typical settings) trigger recalculation
- Scaling path: Already implemented auto-expansion in `pedantic_parakeet/parakeet_mlx/attention.py:572-574`

## Dependencies at Risk

**mlx-audio Optional Dependency:**
- Risk: mlx-audio backend is optional but error message if missing could be clearer
- Impact: Users selecting Whisper or Voxtral models without mlx-audio installed get runtime error
- Migration plan: Consider making mlx-audio a required dependency or improving error UX

**librosa Dependency:**
- Risk: Heavy dependency (brings numpy, scipy, etc.) for audio loading
- Impact: Slow install; potential version conflicts
- Migration plan: Could use lighter audio loading (e.g., soundfile) if librosa features not needed

## Missing Critical Features

**No Real-Time Streaming API:**
- Problem: StreamingParakeet exists but no public API to push raw audio samples
- Blocks: Live microphone transcription use cases
- Files: `pedantic_parakeet/parakeet_mlx/parakeet.py:1090-1097` (add_audio method exists internally)

**No Word-Level Confidence in Output Formats:**
- Problem: SRT/VTT formatters don't expose token-level confidence scores
- Blocks: Highlighting low-confidence words for manual review
- Files: `pedantic_parakeet/formatters.py:76-108`

**No Batch File Processing Progress:**
- Problem: When processing multiple files, no way to track overall progress
- Blocks: Good UX for large batch jobs
- Files: `pedantic_parakeet/cli.py:232-263` (progress only shows file-by-file)

## Test Coverage Gaps

**No Integration Tests:**
- What's not tested: End-to-end transcription with real audio files
- Files: All tests use mocks or synthetic data
- Risk: Model loading, audio processing, and output generation not verified together
- Priority: Medium - would catch integration issues early

**No Streaming Tests:**
- What's not tested: `StreamingParakeet` class and `transcribe_stream()` method
- Files: `pedantic_parakeet/parakeet_mlx/parakeet.py:1012-1199`
- Risk: Streaming could silently break; state management bugs undetected
- Priority: High - streaming is a key feature

**No mlx-audio Backend Tests:**
- What's not tested: `MlxAudioBackend` class with actual mlx-audio library
- Files: `pedantic_parakeet/backends/mlx_audio.py`
- Risk: Backend could fail with real models; output format handling untested
- Priority: Medium - requires mlx-audio to be installed

**Limited Error Path Testing:**
- What's not tested: Error handling for corrupt audio, network failures during model download, invalid model IDs
- Files: Throughout CLI and transcriber
- Risk: Poor error messages; silent failures
- Priority: Low - happy path works

**No Performance Regression Tests:**
- What's not tested: Transcription speed benchmarks
- Files: N/A
- Risk: Performance regressions could go unnoticed
- Priority: Low - not critical for correctness

## Broad Exception Handling

**CLI Exception Swallowing:**
- Issue: Generic `except Exception as e` in file processing
- Files: `pedantic_parakeet/cli.py:297`
- Impact: May hide root cause of failures; loses stack traces
- Fix: Log full traceback in verbose mode; catch specific exceptions

**Vendored CLI Exception Handling:**
- Issue: Multiple broad exception handlers in vendored CLI
- Files: `pedantic_parakeet/parakeet_mlx/cli.py:274`, `pedantic_parakeet/parakeet_mlx/cli.py:394`, `pedantic_parakeet/parakeet_mlx/cli.py:553`, `pedantic_parakeet/parakeet_mlx/cli.py:559`
- Impact: Debugging difficult; may mask bugs
- Fix: Add specific exception types; preserve stack traces

**Utils Silent Failure:**
- Issue: Bare `except Exception:` with pass
- Files: `pedantic_parakeet/parakeet_mlx/utils.py:75`
- Impact: Unknown failures silently ignored
- Fix: At minimum log the exception; determine if it's safe to ignore

---

*Concerns audit: 2026-01-19*
