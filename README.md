# Pedantic Parakeet

*The pedantic parakeet will record your words exactly as spoken, including all the 'ums' and 'ahs' and that thing you said about your mother-in-law that you really shouldn't have.*

A CLI tool for transcribing audio files using NVIDIA Parakeet TDT models via `parakeet-mlx`, optimized for Apple Silicon.

## Installation

```bash
# Clone and install
git clone https://github.com/yourusername/pedantic-parakeet
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
