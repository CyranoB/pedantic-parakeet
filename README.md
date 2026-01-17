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

## License

MIT
