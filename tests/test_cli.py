"""Tests for cli.py formatting functions."""

import pytest

from pedantic_parakeet.parakeet_mlx.alignment import (
    AlignedResult,
    AlignedSentence,
    AlignedToken,
)
from pedantic_parakeet.parakeet_mlx.cli import (
    format_timestamp,
    to_json,
    to_srt,
    to_txt,
    to_vtt,
)


# Helper to create test data
def make_token(id: int, text: str, start: float, duration: float, confidence: float = 1.0) -> AlignedToken:
    return AlignedToken(id=id, text=text, start=start, duration=duration, confidence=confidence)


def make_simple_result() -> AlignedResult:
    """Create a simple result with one sentence."""
    tokens = [
        make_token(1, "Hello", 0.0, 0.5, confidence=0.95),
        make_token(2, " world", 0.5, 0.5, confidence=0.90),
    ]
    sentence = AlignedSentence(text="Hello world", tokens=tokens)
    return AlignedResult(text="Hello world", sentences=[sentence])


def make_multi_sentence_result() -> AlignedResult:
    """Create a result with multiple sentences."""
    tokens1 = [
        make_token(1, "Hello", 0.0, 0.5),
        make_token(2, " world", 0.5, 0.5),
        make_token(3, ".", 1.0, 0.1),
    ]
    tokens2 = [
        make_token(4, " How", 1.5, 0.3),
        make_token(5, " are", 1.8, 0.3),
        make_token(6, " you", 2.1, 0.3),
        make_token(7, "?", 2.4, 0.1),
    ]
    sentence1 = AlignedSentence(text="Hello world.", tokens=tokens1)
    sentence2 = AlignedSentence(text=" How are you?", tokens=tokens2)
    return AlignedResult(text="Hello world. How are you?", sentences=[sentence1, sentence2])


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_zero_seconds(self):
        result = format_timestamp(0.0)
        assert result == "00:00:00,000"

    def test_simple_seconds(self):
        result = format_timestamp(5.0)
        assert result == "00:00:05,000"

    def test_minutes(self):
        result = format_timestamp(65.0)
        assert result == "00:01:05,000"

    def test_hours(self):
        result = format_timestamp(3665.0)  # 1 hour, 1 minute, 5 seconds
        assert result == "01:01:05,000"

    def test_milliseconds(self):
        result = format_timestamp(1.234)
        assert result == "00:00:01,234"

    def test_decimal_marker_period(self):
        result = format_timestamp(1.5, decimal_marker=".")
        assert result == "00:00:01.500"

    def test_without_hours(self):
        result = format_timestamp(65.0, always_include_hours=False)
        assert result == "01:05,000"

    def test_shows_hours_when_needed(self):
        result = format_timestamp(3665.0, always_include_hours=False)
        assert result == "01:01:05,000"


class TestToTxt:
    """Tests for to_txt function."""

    def test_returns_stripped_text(self):
        result = make_simple_result()
        txt = to_txt(result)
        assert txt == "Hello world"

    def test_multi_sentence(self):
        result = make_multi_sentence_result()
        txt = to_txt(result)
        assert "Hello world" in txt
        assert "How are you" in txt


class TestToSrt:
    """Tests for to_srt function."""

    def test_basic_format(self):
        result = make_simple_result()
        srt = to_srt(result)
        
        lines = srt.strip().split("\n")
        assert lines[0] == "1"
        assert "-->" in lines[1]
        assert "Hello world" in lines[2]

    def test_timestamp_format(self):
        result = make_simple_result()
        srt = to_srt(result)
        
        # Should use comma as decimal separator
        assert ",000" in srt or ",500" in srt

    def test_multi_sentence(self):
        result = make_multi_sentence_result()
        srt = to_srt(result)
        
        # Should have entry numbers 1 and 2
        assert "\n1\n" in srt or srt.startswith("1\n")
        assert "\n2\n" in srt

    def test_highlight_words(self):
        result = make_simple_result()
        srt = to_srt(result, highlight_words=True)
        
        # Should contain underline tags
        assert "<u>" in srt
        assert "</u>" in srt

    def test_highlight_words_creates_entry_per_token(self):
        result = make_simple_result()
        srt = to_srt(result, highlight_words=True)
        
        # Should have 2 entries (one per token)
        lines = srt.strip().split("\n\n")
        assert len(lines) == 2


class TestToVtt:
    """Tests for to_vtt function."""

    def test_starts_with_webvtt(self):
        result = make_simple_result()
        vtt = to_vtt(result)
        assert vtt.startswith("WEBVTT")

    def test_timestamp_format(self):
        result = make_simple_result()
        vtt = to_vtt(result)
        
        # Should use period as decimal separator
        assert ".000" in vtt or ".500" in vtt

    def test_basic_format(self):
        result = make_simple_result()
        vtt = to_vtt(result)
        
        assert "-->" in vtt
        assert "Hello world" in vtt

    def test_multi_sentence(self):
        result = make_multi_sentence_result()
        vtt = to_vtt(result)
        
        # Should contain both sentences
        assert "Hello world" in vtt
        assert "How are you" in vtt

    def test_highlight_words(self):
        result = make_simple_result()
        vtt = to_vtt(result, highlight_words=True)
        
        # Should contain bold tags
        assert "<b>" in vtt
        assert "</b>" in vtt

    def test_highlight_words_creates_entry_per_token(self):
        result = make_simple_result()
        vtt = to_vtt(result, highlight_words=True)
        
        # Should have multiple timestamp entries
        assert vtt.count("-->") == 2  # one per token


class TestToJson:
    """Tests for to_json function."""

    def test_returns_valid_json(self):
        import json
        result = make_simple_result()
        json_str = to_json(result)
        
        # Should parse without error
        data = json.loads(json_str)
        assert "text" in data
        assert "sentences" in data

    def test_contains_text(self):
        import json
        result = make_simple_result()
        data = json.loads(to_json(result))
        
        assert data["text"] == "Hello world"

    def test_contains_sentences(self):
        import json
        result = make_multi_sentence_result()
        data = json.loads(to_json(result))
        
        assert len(data["sentences"]) == 2

    def test_sentence_has_tokens(self):
        import json
        result = make_simple_result()
        data = json.loads(to_json(result))
        
        sentence = data["sentences"][0]
        assert "tokens" in sentence
        assert len(sentence["tokens"]) == 2

    def test_token_has_required_fields(self):
        import json
        result = make_simple_result()
        data = json.loads(to_json(result))
        
        token = data["sentences"][0]["tokens"][0]
        assert "text" in token
        assert "start" in token
        assert "end" in token
        assert "duration" in token
        assert "confidence" in token

    def test_values_are_rounded(self):
        import json
        result = make_simple_result()
        data = json.loads(to_json(result))
        
        token = data["sentences"][0]["tokens"][0]
        # Values should be rounded to 3 decimal places
        assert token["start"] == 0.0
        assert token["confidence"] == 0.95


class TestEdgeCases:
    """Edge case tests for formatting functions."""

    def test_empty_sentences(self):
        tokens = [make_token(1, "", 0.0, 0.1)]
        sentence = AlignedSentence(text="", tokens=tokens)
        result = AlignedResult(text="", sentences=[sentence])
        
        # Should not crash
        to_txt(result)
        to_srt(result)
        to_vtt(result)
        to_json(result)

    def test_long_duration(self):
        # 2 hours
        tokens = [make_token(1, "Long", 0.0, 7200.0)]
        sentence = AlignedSentence(text="Long", tokens=tokens)
        result = AlignedResult(text="Long", sentences=[sentence])
        
        srt = to_srt(result)
        # Should handle hours properly
        assert "02:00:00" in srt

    def test_unicode_text(self):
        tokens = [make_token(1, "Привет мир", 0.0, 1.0)]
        sentence = AlignedSentence(text="Привет мир", tokens=tokens)
        result = AlignedResult(text="Привет мир", sentences=[sentence])
        
        txt = to_txt(result)
        assert "Привет мир" in txt
        
        json_out = to_json(result)
        assert "Привет мир" in json_out

    def test_special_characters(self):
        tokens = [make_token(1, "Hello <world> & \"friends\"", 0.0, 1.0)]
        sentence = AlignedSentence(text="Hello <world> & \"friends\"", tokens=tokens)
        result = AlignedResult(text="Hello <world> & \"friends\"", sentences=[sentence])
        
        # Should not escape characters inappropriately
        txt = to_txt(result)
        assert "<world>" in txt
