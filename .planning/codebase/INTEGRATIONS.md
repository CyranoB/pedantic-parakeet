# External Integrations

**Analysis Date:** 2026-01-19

## APIs & External Services

**HuggingFace Hub:**
- Purpose: Model weights and configuration downloading
- SDK/Client: `huggingface-hub` package
- Auth: Not required (uses public models)
- Models fetched:
  - `mlx-community/parakeet-tdt-0.6b-v3` (default)
  - `mlx-community/parakeet-tdt-0.6b-v2`
  - `mlx-community/whisper-large-v3-turbo-asr-fp16`
  - `mlx-community/Voxtral-Mini-3B-2507-bf16`
- Cache location: `~/.cache/huggingface/hub/` (default)
- Implementation: `pedantic_parakeet/parakeet_mlx/utils.py` lines 59-87

**No Other External APIs:**
- No payment providers
- No analytics services
- No telemetry
- No cloud services

## Data Storage

**Databases:**
- None - Stateless CLI tool

**File Storage:**
- Local filesystem only
- Input: Audio files (wav, mp3, m4a, flac, ogg, webm, aac, wma)
- Output: Transcription files (txt, srt, vtt, json)
- Model cache: HuggingFace default cache directory

**Caching:**
- HuggingFace Hub model cache (automatic)
- LRU cache for audio window functions (`pedantic_parakeet/parakeet_mlx/audio.py`)
- No application-level caching

## Authentication & Identity

**Auth Provider:**
- None - No authentication required
- HuggingFace models are public, no API key needed

## Monitoring & Observability

**Error Tracking:**
- None - CLI tool uses stderr for errors

**Logs:**
- Rich console output via `rich.console.Console`
- Verbose mode available via `--verbose` flag
- Progress bars for multi-file processing

**Metrics:**
- None

## CI/CD & Deployment

**Hosting:**
- Not applicable - Local CLI tool
- Distributed via pip install

**CI Pipeline:**
- Not detected in repository

**Release Process:**
- Not configured

## Environment Configuration

**Required env vars:**
- None

**Optional env vars:**
- `HF_HOME` - Override HuggingFace cache directory
- `HF_HUB_CACHE` - Override model cache location

**Secrets location:**
- Not applicable - No secrets required

## External Tools (Runtime Dependencies)

**ffmpeg (Required):**
- Purpose: Audio file decoding and format conversion
- Check: `shutil.which("ffmpeg")` in `pedantic_parakeet/audio.py`
- Usage: `pedantic_parakeet/parakeet_mlx/audio.py` lines 51-74
- Converts any audio format to 16-bit PCM at specified sample rate
- Must be installed and in PATH
- Installation: `brew install ffmpeg` (macOS)

**ffprobe (Optional):**
- Purpose: Get audio duration without loading full file
- Check: `shutil.which("ffprobe")` in `pedantic_parakeet/audio.py`
- Falls back gracefully if not available

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Model Registry

**Supported Models (`pedantic_parakeet/backends/registry.py`):**

| Model ID | Backend | Timestamps | Language Bias | Language Hint |
|----------|---------|------------|---------------|---------------|
| `mlx-community/parakeet-tdt-0.6b-v3` | parakeet | Yes | Yes | No |
| `mlx-community/parakeet-tdt-0.6b-v2` | mlx-audio | Yes | No | No |
| `mlx-community/whisper-large-v3-turbo-asr-fp16` | mlx-audio | Yes | No | Yes |
| `mlx-community/Voxtral-Mini-3B-2507-bf16` | mlx-audio | No | No | Yes |

**Model Aliases:**
- `parakeet`, `parakeet-v3` -> `mlx-community/parakeet-tdt-0.6b-v3`
- `parakeet-v2` -> `mlx-community/parakeet-tdt-0.6b-v2`
- `whisper`, `whisper-turbo` -> `mlx-community/whisper-large-v3-turbo-asr-fp16`
- `voxtral`, `voxtral-mini` -> `mlx-community/Voxtral-Mini-3B-2507-bf16`

## Backend Dependencies

**Parakeet Backend (default):**
- Uses vendored `parakeet_mlx` implementation
- Location: `pedantic_parakeet/parakeet_mlx/`
- No external dependencies beyond core

**MLX-Audio Backend (optional):**
- Requires: `pip install pedantic-parakeet[mlx-audio]`
- Package: `mlx-audio>=0.2.5`
- Lazy-loaded to avoid import errors when not installed
- Check: `_check_mlx_audio_available()` in `pedantic_parakeet/backends/mlx_audio.py`

## Network Requirements

**Model Download:**
- First run downloads model weights (~600MB-3GB depending on model)
- Subsequent runs use cached weights
- No network required after initial download

**Runtime:**
- No network access required during transcription
- All inference runs locally on Apple Silicon

---

*Integration audit: 2026-01-19*
