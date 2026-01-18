"""Language bias for transcription."""

import warnings

import mlx.core as mx
import numpy as np

SUPPORTED_LANGUAGES = frozenset({"fr"})

ENGLISH_SUPPRESS = frozenset({
    "▁the", "▁The", "▁is", "▁are", "▁was", "▁were",
    "▁okay", "▁Okay", "▁OK", "▁so", "▁So",
    "▁and", "▁And", "▁but", "▁But", "▁or", "▁Or",
    "▁it", "▁It", "▁this", "▁This", "▁that", "▁That",
    "▁I", "▁you", "▁You", "▁we", "▁We", "▁they", "▁They",
    "▁have", "▁has", "▁had", "▁do", "▁does", "▁did",
    "▁will", "▁would", "▁could", "▁should",
    "▁what", "▁What", "▁how", "▁How", "▁why", "▁Why",
})


def build_language_bias(
    vocabulary: list[str],
    target_lang: str,
    strength: float = 0.5,
) -> mx.array:
    """
    Build bias vector for target language.

    Args:
        vocabulary: Model vocabulary list
        target_lang: Target language code (must be in SUPPORTED_LANGUAGES)
        strength: Bias strength (0.0-2.0)

    Returns:
        Bias array of shape [vocab_size + 1]

    Raises:
        ValueError: If target_lang not supported or strength out of range
    """
    if target_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Language '{target_lang}' not supported. Use: {sorted(SUPPORTED_LANGUAGES)}"
        )

    if not 0.0 <= strength <= 2.0:
        raise ValueError(f"strength must be 0.0-2.0, got {strength}")

    vocab_size = len(vocabulary)
    bias_np = np.zeros(vocab_size + 1, dtype=np.float32)

    if target_lang == "fr":
        suppressed = 0
        for i, token in enumerate(vocabulary):
            if token in ENGLISH_SUPPRESS:
                bias_np[i] = -strength
                suppressed += 1
        if suppressed == 0:
            warnings.warn("No English tokens found to suppress")

    return mx.array(bias_np)
