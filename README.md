![Pedantic Parakeet](assets/header.png)

# Pedantic Parakeet

*In the same way that the Guild of Accountants believes every penny must be accounted for, the pedantic parakeet believes every syllable deserves its moment in the permanent record—a philosophy that sounds admirable right up until you remember what you said to the girl from HR at the Christmas party after your fourth glass of wine.*

A CLI tool for transcribing audio files using NVIDIA Parakeet TDT models via `parakeet-mlx`, optimized for Apple Silicon.

## Installation

```bash
# Clone and install
git clone https://github.com/CyranoB/pedantic-parakeet
cd pedantic-parakeet
uv sync
```

Or install from PyPI (coming soon):
```bash
pip install pedantic-parakeet
```

**System requirement**: `ffmpeg` must be installed (`brew install ffmpeg`)

## Usage

```bash
# Basic usage
pedantic-parakeet audio.mp3

# Batch processing
pedantic-parakeet ./lectures/ --output ./transcripts/

# Output formats
pedantic-parakeet audio.mp3 --format srt
pedantic-parakeet audio.mp3 --format txt,srt,vtt,json

# With timestamps in plain text
pedantic-parakeet audio.mp3 --format txt --timestamps

# Preview what would be processed
pedantic-parakeet ./lectures/ --dry-run

# Verbose mode
pedantic-parakeet ./lectures/ -v
```

## Language Bias (Experimental)

### The Problem

The decoder can drift into English for short stretches. When a French speaker uses common English expressions like "okay" or "so", the model may:
- Transcribe that segment entirely in English
- Switch mid-sentence between languages
- Produce inconsistent output for the same speaker

This is especially common with:
- **Bilingual speakers** who naturally mix languages
- **Canadian French** speakers who use English expressions ("by the way", "anyway", "you know")
- **Technical content** where English terms are standard
- **Casual speech** with fillers like "okay", "so", "well"

### The Solution

The `--language` flag biases decoding away from a small set of common English filler words and expressions. This helps reduce code-switching when the speaker uses occasional English words.

```bash
# Basic usage - transcribe French audio
pedantic-parakeet lecture.mp3 --language fr
```

### Understanding Strength

The `--language-strength` parameter (0.0-2.0) controls how aggressively English words are suppressed:

| Strength | Effect | Best for |
|----------|--------|----------|
| `0.0` | No suppression (same as not using --language) | Testing/comparison |
| `0.5` | Light suppression (default) | Content with intentional English quotes or terms |
| `1.0` | Moderate suppression | Most French content with occasional anglicisms |
| `1.5` | Strong suppression | Canadian French or heavily code-switched speech |
| `2.0` | Maximum suppression | When you want minimal English regardless of context |

**Important**: Higher strength values may cause the model to substitute phonetically similar French words for intentional English. For example, "meeting" might become "mitaine" (mitten). Start with the default and increase only if needed.

### Examples

```bash
# European French lecture (occasional "okay" or "so")
pedantic-parakeet lecture.mp3 --language fr

# Canadian French podcast (frequent English expressions)
pedantic-parakeet podcast.mp3 --language fr --language-strength 1.5

# Interview where English quotes should be preserved
pedantic-parakeet interview.mp3 --language fr --language-strength 0.3
```

### What Gets Suppressed

For French (`--language fr`), the following English words are suppressed:
- Fillers: okay, OK, so
- Conjunctions: and, but, or
- Articles/pronouns: the, it, this, that, I, you, we, they
- Common verbs: is, are, was, were, have, has, had, do, does, did, will, would, could, should
- Question words: what, how, why

Suppression is token-level, so multi-word phrases like "by the way" are not explicitly targeted.
Words that are intentionally English (proper nouns, technical terms, quotes) may still come through, especially at lower strength values.

### Currently Supported Languages

- `fr` - French (suppresses common English words)

More languages may be added in future versions.

## Output Formats

- **txt**: Plain text transcript (optionally with timestamps)
- **srt**: SubRip subtitle format
- **vtt**: WebVTT subtitle format
- **json**: Structured JSON with timestamps and confidence scores

## Features

- Batch processing of audio files and directories
- Automatic audio format conversion (mp3, m4a, wav, flac, ogg, webm)
- Multiple output formats in a single run
- Progress bars for long transcriptions
- Chunked processing for long audio files

## Model

Uses NVIDIA's Parakeet TDT models via MLX, optimized for Apple Silicon. The default model (`parakeet-tdt-0.6b-v3`) will be downloaded on first run (~1GB).

## Notes

- The decoder can drift into English in short stretches. See [Language Bias](#language-bias-experimental) if you're transcribing non-English content with occasional English words.

## License

MIT
