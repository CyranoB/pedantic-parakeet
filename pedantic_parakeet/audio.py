"""Audio file discovery and validation utilities."""

import shutil
import subprocess
from pathlib import Path

# Formats supported by librosa/ffmpeg (used by parakeet-mlx internally)
SUPPORTED_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".wma"
})


def is_supported_audio(path: Path) -> bool:
    """Check if a file has a supported audio extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_audio_files(paths: list[Path], recursive: bool = False) -> list[Path]:
    """
    Discover audio files from a list of paths.

    Args:
        paths: List of file or directory paths
        recursive: If True, search directories recursively

    Returns:
        List of audio file paths, sorted alphabetically
    """
    audio_files: list[Path] = []

    for path in paths:
        if path.is_file():
            if is_supported_audio(path):
                audio_files.append(path)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for file_path in path.glob(pattern):
                if file_path.is_file() and is_supported_audio(file_path):
                    audio_files.append(file_path)

    # Sort for consistent ordering
    return sorted(set(audio_files))


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def get_audio_duration(path: Path) -> float | None:
    """
    Get audio duration in seconds using ffprobe.

    Returns None if ffprobe is not available or fails.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass

    return None
