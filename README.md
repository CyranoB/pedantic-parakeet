![Pedantic Parakeet](assets/header.png)

# Pedantic Parakeet

*In the same way that the Guild of Accountants believes every penny must be accounted for, the pedantic parakeet believes every syllable deserves its moment in the permanent record—a philosophy that sounds admirable right up until you remember what you said to the girl from HR at the Christmas party after your fourth glass of wine.*

A CLI tool for transcribing audio files using MLX models optimized for Apple Silicon. Supports multiple backends including NVIDIA Parakeet TDT, OpenAI Whisper, and Mistral Voxtral.

## Features

- **Multiple backends** — Parakeet (default), Whisper, Voxtral via mlx-audio
- **Multiple output formats** — txt, srt, vtt, json in a single run
- **Batch processing** — Process entire directories recursively
- **Language bias** — Reduce code-switching for non-English content
- **Smart validation** — Clear errors when model capabilities don't match requested options

## Installation

```bash
# Clone and install
git clone https://github.com/CyranoB/pedantic-parakeet
cd pedantic-parakeet
uv sync
```

Or install from PyPI:
```bash
pip install pedantic-parakeet
```

**Optional: mlx-audio backend** (for Whisper and Voxtral models):
```bash
pip install pedantic-parakeet[mlx-audio]
```

**System requirement**: `ffmpeg` must be installed (`brew install ffmpeg`)

## Quick Start

```bash
# Transcribe a single file (outputs SRT by default)
pedantic-parakeet audio.mp3

# Output plain text
pedantic-parakeet audio.mp3 --format txt

# Multiple formats at once
pedantic-parakeet audio.mp3 --format txt,srt,json

# Process a directory
pedantic-parakeet ./recordings/ --output ./transcripts/

# Preview what would be processed
pedantic-parakeet ./recordings/ --dry-run
```

## Available Models

List all supported models with their capabilities:

```bash
pedantic-parakeet --list-models
```

```
Supported Models:

  mlx-community/parakeet-tdt-0.6b-v3
    Backend: parakeet | Timestamps: ✓ | Aliases: parakeet-v3, parakeet
    Parakeet TDT 0.6B v3 - High accuracy, language bias support

  mlx-community/whisper-large-v3-turbo-asr-fp16
    Backend: mlx-audio | Timestamps: ✓ | Aliases: whisper-turbo, whisper
    Whisper Large v3 Turbo - Fast, 100+ languages

  mlx-community/Voxtral-Mini-3B-2507-bf16
    Backend: mlx-audio | Timestamps: ✗ | Aliases: voxtral, voxtral-mini
    Voxtral Mini 3B - LLM-based, context-aware transcription
```

### Model Comparison

| Model | Backend | Timestamps | Best For |
|-------|---------|------------|----------|
| **parakeet** (default) | parakeet | ✓ | English, high accuracy, language bias |
| **whisper-turbo** | mlx-audio | ✓ | Multilingual (100+ languages) |
| **voxtral** | mlx-audio | ✗ | Context-aware, LLM-based (text only) |

### Selecting a Model

```bash
# Use default Parakeet model
pedantic-parakeet audio.mp3

# Use model alias
pedantic-parakeet audio.mp3 --model whisper

# Use full model ID
pedantic-parakeet audio.mp3 --model mlx-community/whisper-large-v3-turbo-asr-fp16

# Explicitly select backend
pedantic-parakeet audio.mp3 --backend mlx-audio --model whisper
```

### Important: Voxtral and Timestamps

Voxtral does not provide word-level timestamps. If you request subtitle formats (srt, vtt, json) with Voxtral, you'll get a clear error:

```
Error: Model 'voxtral' does not support timestamps.
Cannot use formats: srt. Use --format txt instead, or choose a different model.
```

Use `--format txt` with Voxtral:
```bash
pedantic-parakeet audio.mp3 --model voxtral --format txt
```

## Output Formats

| Format | Description | Requires Timestamps |
|--------|-------------|---------------------|
| **srt** | SubRip subtitles (default) | Yes |
| **vtt** | WebVTT subtitles | Yes |
| **json** | Structured data with confidence scores | Yes |
| **txt** | Plain text | No |

```bash
# Plain text with timestamps
pedantic-parakeet audio.mp3 --format txt --timestamps

# All formats at once
pedantic-parakeet audio.mp3 --format all
```

## Language Bias (Parakeet Only)

When transcribing non-English audio, the Parakeet model may occasionally produce English words for common fillers ("okay", "so", "the"). The `--language` flag biases decoding away from these words.

```bash
# French audio with occasional English expressions
pedantic-parakeet lecture.mp3 --language fr

# Stronger bias for heavily code-switched speech
pedantic-parakeet podcast.mp3 --language fr --language-strength 1.5
```

### Strength Levels

| Strength | Effect | Use Case |
|----------|--------|----------|
| 0.0 | No suppression | Testing/comparison |
| 0.5 | Light (default) | Preserve intentional English |
| 1.0 | Moderate | Most non-English content |
| 1.5 | Strong | Canadian French, heavy code-switching |
| 2.0 | Maximum | Minimal English regardless of context |

**Note:** Language bias only works with the Parakeet backend. Using `--language-strength` with Whisper or Voxtral will produce an error:

```
Error: Model 'whisper' does not support --language-strength.
Models with language bias support: mlx-community/parakeet-tdt-0.6b-v3
```

**Currently supported:** French (`fr`)

## CLI Reference

```
pedantic-parakeet [OPTIONS] INPUTS...

Arguments:
  INPUTS...              Audio files or directories to transcribe

Options:
  -o, --output PATH      Output directory (default: same as input)
  -f, --format TEXT      Output format(s): txt, srt, vtt, json, or 'all'
  -r, --recursive        Search directories recursively
  -t, --timestamps       Include timestamps in plain text output
  --dry-run              Show what would be processed without transcribing
  -m, --model TEXT       Model ID or alias (see --list-models)
  -b, --backend TEXT     Backend: parakeet or mlx-audio (auto-detected)
  --list-models          List supported models and exit
  --chunk-duration FLOAT Chunk duration for long audio (default: 120s)
  -l, --language TEXT    Target language for bias (fr)
  --language-strength    Bias strength 0.0-2.0 (default: 0.5)
  -v, --verbose          Show detailed progress
  -V, --version          Show version and exit
```

## Examples

### Basic Transcription

```bash
# Single file to SRT
pedantic-parakeet meeting.mp3

# Single file to text
pedantic-parakeet meeting.mp3 -f txt

# With verbose progress
pedantic-parakeet meeting.mp3 -v
```

### Batch Processing

```bash
# Process directory
pedantic-parakeet ./recordings/

# Recursive with custom output
pedantic-parakeet ./recordings/ -r --output ./transcripts/

# Preview first
pedantic-parakeet ./recordings/ -r --dry-run
```

### Multilingual Content

```bash
# Use Whisper for non-English languages
pedantic-parakeet spanish_audio.mp3 --model whisper

# French with language bias (Parakeet)
pedantic-parakeet french_lecture.mp3 --language fr
```

### Multiple Formats

```bash
# Text and subtitles
pedantic-parakeet audio.mp3 -f txt,srt

# All formats
pedantic-parakeet audio.mp3 -f all
```

## Notes

- Models are downloaded on first use (~1-3GB depending on model)
- Parakeet models are optimized for English but work with other languages
- For best multilingual support, use Whisper
- Long audio files are automatically chunked (configurable with `--chunk-duration`)

## License

MIT
