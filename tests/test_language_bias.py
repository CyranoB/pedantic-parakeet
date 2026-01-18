"""Tests for language bias functionality."""

import pytest

from pedantic_parakeet.language_bias import (
    ENGLISH_SUPPRESS,
    SUPPORTED_LANGUAGES,
    build_language_bias,
)


def test_bias_shape():
    """Bias array should have shape [vocab_size + 1]."""
    vocab = ["a", "b", "c", "▁the"]
    bias = build_language_bias(vocab, "fr", 0.5)
    assert bias.shape == (5,)


def test_english_suppression():
    """English tokens should have negative bias."""
    vocab = ["▁le", "▁the", "▁la"]
    bias = build_language_bias(vocab, "fr", 0.5)
    # "▁the" is at index 1 and should be suppressed
    assert abs(float(bias[1]) - (-0.5)) < 1e-6
    # French words should not be suppressed
    assert abs(float(bias[0]) - 0.0) < 1e-6
    assert abs(float(bias[2]) - 0.0) < 1e-6


def test_bias_strength():
    """Bias strength should be applied correctly."""
    vocab = ["▁the", "▁is"]
    bias_05 = build_language_bias(vocab, "fr", 0.5)
    bias_10 = build_language_bias(vocab, "fr", 1.0)

    assert abs(float(bias_05[0]) - (-0.5)) < 1e-6
    assert abs(float(bias_10[0]) - (-1.0)) < 1e-6


def test_unsupported_language_raises():
    """Unsupported language should raise ValueError."""
    with pytest.raises(ValueError, match="not supported"):
        build_language_bias(["a"], "xx", 0.5)


def test_invalid_strength_raises_negative():
    """Negative strength should raise ValueError."""
    with pytest.raises(ValueError, match="strength must be"):
        build_language_bias(["a"], "fr", -1.0)


def test_invalid_strength_raises_too_high():
    """Strength > 2.0 should raise ValueError."""
    with pytest.raises(ValueError, match="strength must be"):
        build_language_bias(["a"], "fr", 5.0)


def test_boundary_strength_values():
    """Boundary values (0.0 and 2.0) should be accepted."""
    vocab = ["▁the"]
    # Should not raise
    bias_zero = build_language_bias(vocab, "fr", 0.0)
    bias_max = build_language_bias(vocab, "fr", 2.0)

    assert abs(float(bias_zero[0]) - 0.0) < 1e-6
    assert abs(float(bias_max[0]) - (-2.0)) < 1e-6


def test_supported_languages():
    """SUPPORTED_LANGUAGES should contain expected languages."""
    assert "fr" in SUPPORTED_LANGUAGES


def test_english_suppress_tokens():
    """ENGLISH_SUPPRESS should contain common English tokens."""
    assert "▁the" in ENGLISH_SUPPRESS
    assert "▁The" in ENGLISH_SUPPRESS
    assert "▁okay" in ENGLISH_SUPPRESS


def test_no_english_tokens_warns():
    """Should warn when no English tokens found to suppress."""
    vocab = ["▁le", "▁la", "▁de"]  # No English tokens
    with pytest.warns(UserWarning, match="No English tokens found"):
        build_language_bias(vocab, "fr", 0.5)
