"""Tests for alignment.py functions."""

import pytest

from pedantic_parakeet.parakeet_mlx.alignment import (
    AlignedResult,
    AlignedSentence,
    AlignedToken,
    SentenceConfig,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)


# Helper to create tokens easily
def make_token(id: int, text: str, start: float, duration: float, confidence: float = 1.0) -> AlignedToken:
    return AlignedToken(id=id, text=text, start=start, duration=duration, confidence=confidence)


class TestAlignedToken:
    """Tests for AlignedToken dataclass."""

    def test_end_computed_from_start_and_duration(self):
        token = make_token(1, "hello", start=1.0, duration=0.5)
        assert token.end == pytest.approx(1.5)

    def test_default_confidence(self):
        token = AlignedToken(id=1, text="test", start=0.0, duration=1.0)
        assert token.confidence == pytest.approx(1.0)


class TestAlignedSentence:
    """Tests for AlignedSentence dataclass."""

    def test_sentence_computes_start_end_duration(self):
        tokens = [
            make_token(1, "Hello", 0.0, 0.5),
            make_token(2, " world", 0.5, 0.5),
        ]
        sentence = AlignedSentence(text="Hello world", tokens=tokens)
        assert sentence.start == pytest.approx(0.0)
        assert sentence.end == pytest.approx(1.0)
        assert sentence.duration == pytest.approx(1.0)

    def test_sentence_sorts_tokens_by_start(self):
        tokens = [
            make_token(2, " world", 0.5, 0.5),
            make_token(1, "Hello", 0.0, 0.5),
        ]
        sentence = AlignedSentence(text="Hello world", tokens=tokens)
        assert sentence.tokens[0].id == 1
        assert sentence.tokens[1].id == 2

    def test_sentence_computes_confidence(self):
        tokens = [
            make_token(1, "Hello", 0.0, 0.5, confidence=0.9),
            make_token(2, " world", 0.5, 0.5, confidence=0.8),
        ]
        sentence = AlignedSentence(text="Hello world", tokens=tokens)
        # Geometric mean of 0.9 and 0.8
        assert 0.84 < sentence.confidence < 0.86


class TestAlignedResult:
    """Tests for AlignedResult dataclass."""

    def test_result_strips_text(self):
        tokens = [make_token(1, "Hello", 0.0, 0.5)]
        sentence = AlignedSentence(text="Hello", tokens=tokens)
        result = AlignedResult(text="  Hello  ", sentences=[sentence])
        assert result.text == "Hello"

    def test_result_tokens_property(self):
        tokens1 = [make_token(1, "Hello", 0.0, 0.5)]
        tokens2 = [make_token(2, "world", 1.0, 0.5)]
        sentence1 = AlignedSentence(text="Hello", tokens=tokens1)
        sentence2 = AlignedSentence(text="world", tokens=tokens2)
        result = AlignedResult(text="Hello world", sentences=[sentence1, sentence2])
        assert len(result.tokens) == 2
        assert result.tokens[0].id == 1
        assert result.tokens[1].id == 2


class TestTokensToSentences:
    """Tests for tokens_to_sentences function."""

    def test_splits_on_period(self):
        tokens = [
            make_token(1, "Hello", 0.0, 0.5),
            make_token(2, ".", 0.5, 0.1),
            make_token(3, " World", 0.6, 0.5),
        ]
        sentences = tokens_to_sentences(tokens)
        assert len(sentences) == 2

    def test_splits_on_question_mark(self):
        tokens = [
            make_token(1, "Hello", 0.0, 0.5),
            make_token(2, "?", 0.5, 0.1),
            make_token(3, " World", 0.6, 0.5),
        ]
        sentences = tokens_to_sentences(tokens)
        assert len(sentences) == 2

    def test_splits_on_exclamation_mark(self):
        tokens = [
            make_token(1, "Hello", 0.0, 0.5),
            make_token(2, "!", 0.5, 0.1),
            make_token(3, " World", 0.6, 0.5),
        ]
        sentences = tokens_to_sentences(tokens)
        assert len(sentences) == 2

    def test_splits_on_max_words(self):
        tokens = [
            make_token(1, " One", 0.0, 0.5),
            make_token(2, " two", 0.5, 0.5),
            make_token(3, " three", 1.0, 0.5),
            make_token(4, " four", 1.5, 0.5),
        ]
        config = SentenceConfig(max_words=2)
        sentences = tokens_to_sentences(tokens, config)
        assert len(sentences) == 2

    def test_splits_on_silence_gap(self):
        tokens = [
            make_token(1, "Hello", 0.0, 0.5),
            make_token(2, " World", 2.0, 0.5),  # 1.5s gap
        ]
        config = SentenceConfig(silence_gap=1.0)
        sentences = tokens_to_sentences(tokens, config)
        assert len(sentences) == 2

    def test_splits_on_max_duration(self):
        tokens = [
            make_token(1, "Hello", 0.0, 0.5),
            make_token(2, " there", 0.5, 0.5),
            make_token(3, " world", 1.0, 0.5),
        ]
        config = SentenceConfig(max_duration=1.0)
        sentences = tokens_to_sentences(tokens, config)
        assert len(sentences) == 2

    def test_no_split_without_config(self):
        tokens = [
            make_token(1, "Hello", 0.0, 0.5),
            make_token(2, " there", 0.5, 0.5),
            make_token(3, " world", 1.0, 0.5),
        ]
        sentences = tokens_to_sentences(tokens)
        assert len(sentences) == 1


class TestSentencesToResult:
    """Tests for sentences_to_result function."""

    def test_combines_sentences_into_result(self):
        tokens1 = [make_token(1, "Hello", 0.0, 0.5)]
        tokens2 = [make_token(2, " world", 0.5, 0.5)]
        sentence1 = AlignedSentence(text="Hello", tokens=tokens1)
        sentence2 = AlignedSentence(text=" world", tokens=tokens2)
        result = sentences_to_result([sentence1, sentence2])
        assert result.text == "Hello world"
        assert len(result.sentences) == 2


class TestMergeLongestContiguous:
    """Tests for merge_longest_contiguous function."""

    def test_empty_a_returns_b(self):
        b = [make_token(1, "hello", 0.0, 0.5)]
        result = merge_longest_contiguous([], b, overlap_duration=1.0)
        assert result == b

    def test_empty_b_returns_a(self):
        a = [make_token(1, "hello", 0.0, 0.5)]
        result = merge_longest_contiguous(a, [], overlap_duration=1.0)
        assert result == a

    def test_non_overlapping_concatenates(self):
        a = [make_token(1, "hello", 0.0, 0.5)]
        b = [make_token(2, "world", 1.0, 0.5)]
        result = merge_longest_contiguous(a, b, overlap_duration=1.0)
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2

    def test_overlapping_with_matching_tokens(self):
        # Create overlapping sequences with matching token IDs
        a = [
            make_token(1, "one", 0.0, 0.5),
            make_token(2, "two", 0.5, 0.5),
            make_token(3, "three", 1.0, 0.5),  # overlap region
            make_token(4, "four", 1.5, 0.5),   # overlap region
        ]
        b = [
            make_token(3, "three", 1.0, 0.5),  # overlap region - matches
            make_token(4, "four", 1.5, 0.5),   # overlap region - matches
            make_token(5, "five", 2.0, 0.5),
        ]
        result = merge_longest_contiguous(a, b, overlap_duration=2.0)
        # Should merge properly
        assert len(result) >= 4
        # First tokens should be from a
        assert result[0].id == 1
        # Last token should be from b
        assert result[-1].id == 5

    def test_overlapping_without_matching_tokens_raises(self):
        # Overlapping but no matching token IDs - raises RuntimeError
        a = [
            make_token(1, "one", 0.0, 0.5),
            make_token(2, "two", 0.5, 0.5),
        ]
        b = [
            make_token(3, "three", 0.3, 0.5),  # overlaps but different ID
            make_token(4, "four", 0.8, 0.5),
        ]
        with pytest.raises(RuntimeError):
            merge_longest_contiguous(a, b, overlap_duration=1.0)

    def test_raises_when_no_matching_pairs(self):
        # Create scenario where overlap exists but no sufficient matches
        a = [
            make_token(1, "one", 0.0, 0.5),
            make_token(2, "two", 0.5, 0.5),
            make_token(3, "three", 1.0, 0.5),
            make_token(4, "four", 1.5, 0.5),
            make_token(5, "five", 2.0, 0.5),
            make_token(6, "six", 2.5, 0.5),
        ]
        b = [
            make_token(10, "ten", 1.0, 0.5),  # different IDs
            make_token(11, "eleven", 1.5, 0.5),
            make_token(12, "twelve", 2.0, 0.5),
        ]
        with pytest.raises(RuntimeError, match="No pairs"):
            merge_longest_contiguous(a, b, overlap_duration=2.0)


class TestMergeLongestCommonSubsequence:
    """Tests for merge_longest_common_subsequence function."""

    def test_empty_a_returns_b(self):
        b = [make_token(1, "hello", 0.0, 0.5)]
        result = merge_longest_common_subsequence([], b, overlap_duration=1.0)
        assert result == b

    def test_empty_b_returns_a(self):
        a = [make_token(1, "hello", 0.0, 0.5)]
        result = merge_longest_common_subsequence(a, [], overlap_duration=1.0)
        assert result == a

    def test_non_overlapping_concatenates(self):
        a = [make_token(1, "hello", 0.0, 0.5)]
        b = [make_token(2, "world", 1.0, 0.5)]
        result = merge_longest_common_subsequence(a, b, overlap_duration=1.0)
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2

    def test_overlapping_with_matching_tokens(self):
        # Create overlapping sequences with matching token IDs
        a = [
            make_token(1, "one", 0.0, 0.5),
            make_token(2, "two", 0.5, 0.5),
            make_token(3, "three", 1.0, 0.5),  # overlap
            make_token(4, "four", 1.5, 0.5),   # overlap
        ]
        b = [
            make_token(3, "three", 1.0, 0.5),  # matches
            make_token(4, "four", 1.5, 0.5),   # matches
            make_token(5, "five", 2.0, 0.5),
        ]
        result = merge_longest_common_subsequence(a, b, overlap_duration=2.0)
        assert len(result) >= 4
        assert result[0].id == 1
        assert result[-1].id == 5

    def test_overlapping_without_matching_uses_cutoff(self):
        # Overlapping but no matching token IDs - should use cutoff
        a = [
            make_token(1, "one", 0.0, 0.5),
            make_token(2, "two", 0.5, 0.5),
        ]
        b = [
            make_token(3, "three", 0.3, 0.5),
            make_token(4, "four", 0.8, 0.5),
        ]
        result = merge_longest_common_subsequence(a, b, overlap_duration=1.0)
        assert len(result) >= 1

    def test_handles_non_contiguous_matches(self):
        # LCS can handle non-contiguous matching sequences
        a = [
            make_token(1, "one", 0.0, 0.5),
            make_token(2, "two", 0.5, 0.5),
            make_token(3, "three", 1.0, 0.5),
            make_token(99, "extra", 1.3, 0.2),  # extra token in a
            make_token(4, "four", 1.5, 0.5),
        ]
        b = [
            make_token(3, "three", 1.0, 0.5),
            make_token(4, "four", 1.5, 0.5),
            make_token(5, "five", 2.0, 0.5),
        ]
        result = merge_longest_common_subsequence(a, b, overlap_duration=2.0)
        # Should still find the LCS (3, 4)
        assert result[0].id == 1
        assert result[-1].id == 5

    def test_small_overlap_uses_cutoff(self):
        # When overlap region has < 2 tokens in either list, uses cutoff
        a = [make_token(1, "one", 0.0, 0.3)]
        b = [make_token(2, "two", 0.5, 0.5)]  # slight overlap with wider window
        result = merge_longest_common_subsequence(a, b, overlap_duration=0.5)
        # With cutoff at ~0.4, token 1 (ends at 0.3) is kept, token 2 (starts at 0.5) is kept
        assert len(result) == 2


class TestIntegration:
    """Integration tests for the alignment module."""

    def test_full_pipeline(self):
        """Test complete flow: tokens -> sentences -> result."""
        tokens = [
            make_token(1, "Hello", 0.0, 0.5, confidence=0.95),
            make_token(2, " world", 0.5, 0.5, confidence=0.90),
            make_token(3, ".", 1.0, 0.1, confidence=0.99),
            make_token(4, " How", 1.2, 0.3, confidence=0.88),
            make_token(5, " are", 1.5, 0.3, confidence=0.92),
            make_token(6, " you", 1.8, 0.3, confidence=0.91),
            make_token(7, "?", 2.1, 0.1, confidence=0.97),
        ]
        
        sentences = tokens_to_sentences(tokens)
        result = sentences_to_result(sentences)
        
        assert len(result.sentences) == 2
        assert "Hello world" in result.text
        assert "How are you" in result.text
        assert result.sentences[0].confidence > 0
        assert result.sentences[1].confidence > 0

    def test_merge_chunked_audio(self):
        """Simulate merging results from chunked audio processing."""
        # Chunk 1 result
        chunk1 = [
            make_token(1, "The", 0.0, 0.3),
            make_token(2, " quick", 0.3, 0.4),
            make_token(3, " brown", 0.7, 0.4),
            make_token(4, " fox", 1.1, 0.3),
        ]
        
        # Chunk 2 result (overlapping)
        chunk2 = [
            make_token(3, " brown", 0.7, 0.4),  # overlap
            make_token(4, " fox", 1.1, 0.3),    # overlap
            make_token(5, " jumps", 1.4, 0.4),
            make_token(6, " over", 1.8, 0.3),
        ]
        
        merged = merge_longest_common_subsequence(chunk1, chunk2, overlap_duration=1.0)
        
        # Should have all unique tokens
        assert merged[0].id == 1  # "The"
        assert merged[-1].id == 6  # "over"
        # Check no duplicates in IDs for the overlapping tokens
        ids = [t.id for t in merged]
        assert ids.count(3) == 1  # "brown" appears once
        assert ids.count(4) == 1  # "fox" appears once
