from dataclasses import dataclass

import numpy as np


@dataclass
class AlignedToken:
    id: int
    text: str
    start: float
    duration: float
    confidence: float = 1.0  # confidence score (0.0 to 1.0)
    end: float = 0.0  # temporary

    def __post_init__(self) -> None:
        self.end = self.start + self.duration


@dataclass
class AlignedSentence:
    text: str
    tokens: list[AlignedToken]
    start: float = 0.0  # temporary
    end: float = 0.0  # temporary
    duration: float = 0.0  # temporary
    confidence: float = 1.0  # aggregate confidence score

    def __post_init__(self) -> None:
        self.tokens = sorted(self.tokens, key=lambda x: x.start)
        self.start = self.tokens[0].start
        self.end = self.tokens[-1].end
        self.duration = self.end - self.start
        # Compute geometric mean of token confidences
        confidences = np.array([t.confidence for t in self.tokens])
        self.confidence = float(np.exp(np.mean(np.log(confidences + 1e-10))))


@dataclass
class AlignedResult:
    text: str
    sentences: list[AlignedSentence]

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    @property
    def tokens(self) -> list[AlignedToken]:
        return [token for sentence in self.sentences for token in sentence.tokens]


@dataclass
class SentenceConfig:
    max_words: int | None = None
    silence_gap: float | None = None
    max_duration: float | None = None


def tokens_to_sentences(
    tokens: list[AlignedToken], config: SentenceConfig = SentenceConfig()
) -> list[AlignedSentence]:
    sentences = []
    current_tokens: list[AlignedToken] = []

    for idx, token in enumerate(tokens):
        current_tokens.append(token)

        is_punctuation = (
            # hacky, will fix
            "!" in token.text
            or "?" in token.text
            or "。" in token.text
            or "？" in token.text
            or "！" in token.text
            or (
                "." in token.text
                and (idx == len(tokens) - 1 or " " in tokens[idx + 1].text)
            )
        )
        is_word_limit = (
            (config.max_words is not None)
            and (idx != len(tokens) - 1)
            and (
                len([x for x in current_tokens if " " in x.text])
                + (1 if " " in tokens[idx + 1].text else 0)
                > config.max_words
            )
        )
        is_long_silence = (
            (config.silence_gap is not None)
            and (idx != len(tokens) - 1)
            and (tokens[idx + 1].start - token.end >= config.silence_gap)
        )
        is_over_duration = (config.max_duration is not None) and (
            token.end - current_tokens[0].start >= config.max_duration
        )

        if is_punctuation or is_word_limit or is_long_silence or is_over_duration:
            sentence_text = "".join(t.text for t in current_tokens)
            sentence = AlignedSentence(text=sentence_text, tokens=current_tokens)
            sentences.append(sentence)

            current_tokens = []

    if current_tokens:
        sentence_text = "".join(t.text for t in current_tokens)
        sentence = AlignedSentence(text=sentence_text, tokens=current_tokens)
        sentences.append(sentence)

    return sentences


def sentences_to_result(sentences: list[AlignedSentence]) -> AlignedResult:
    return AlignedResult("".join(sentence.text for sentence in sentences), sentences)


def _tokens_match(
    token_a: AlignedToken,
    token_b: AlignedToken,
    max_time_diff: float,
) -> bool:
    """Check if two tokens match by ID and have similar start times."""
    return (
        token_a.id == token_b.id
        and abs(token_a.start - token_b.start) < max_time_diff
    )


def _get_overlap_regions(
    a: list[AlignedToken],
    b: list[AlignedToken],
    overlap_duration: float,
) -> tuple[list[AlignedToken], list[AlignedToken], float, float]:
    """Extract overlapping regions from two token lists."""
    a_end_time = a[-1].end
    b_start_time = b[0].start
    overlap_a = [token for token in a if token.end > b_start_time - overlap_duration]
    overlap_b = [token for token in b if token.start < a_end_time + overlap_duration]
    return overlap_a, overlap_b, a_end_time, b_start_time


def _merge_by_cutoff(
    a: list[AlignedToken],
    b: list[AlignedToken],
    cutoff_time: float,
) -> list[AlignedToken]:
    """Merge two token lists using a time-based cutoff."""
    return [t for t in a if t.end <= cutoff_time] + [
        t for t in b if t.start >= cutoff_time
    ]


def _merge_with_pairs(
    a: list[AlignedToken],
    b: list[AlignedToken],
    pairs: list[tuple[int, int]],
    a_start_idx: int,
) -> list[AlignedToken]:
    """Merge token lists using matched pairs, filling gaps with longer sequence."""
    lcs_indices_a = [a_start_idx + pair[0] for pair in pairs]
    lcs_indices_b = [pair[1] for pair in pairs]

    result: list[AlignedToken] = []
    result.extend(a[: lcs_indices_a[0]])

    for i, (idx_a, idx_b) in enumerate(zip(lcs_indices_a, lcs_indices_b)):
        result.append(a[idx_a])
        if i < len(pairs) - 1:
            next_idx_a = lcs_indices_a[i + 1]
            next_idx_b = lcs_indices_b[i + 1]
            gap_a = a[idx_a + 1 : next_idx_a]
            gap_b = b[idx_b + 1 : next_idx_b]
            result.extend(gap_b if len(gap_b) > len(gap_a) else gap_a)

    result.extend(b[lcs_indices_b[-1] + 1 :])
    return result


def _find_best_contiguous_match(
    overlap_a: list[AlignedToken],
    overlap_b: list[AlignedToken],
    max_time_diff: float,
) -> list[tuple[int, int]]:
    """Find the longest contiguous matching sequence between overlapping regions."""
    best_contiguous: list[tuple[int, int]] = []

    for i in range(len(overlap_a)):
        for j in range(len(overlap_b)):
            if not _tokens_match(overlap_a[i], overlap_b[j], max_time_diff):
                continue

            current = _extend_contiguous_match(
                overlap_a, overlap_b, i, j, max_time_diff
            )
            if len(current) > len(best_contiguous):
                best_contiguous = current

    return best_contiguous


def _extend_contiguous_match(
    overlap_a: list[AlignedToken],
    overlap_b: list[AlignedToken],
    start_i: int,
    start_j: int,
    max_time_diff: float,
) -> list[tuple[int, int]]:
    """Extend a contiguous match from starting indices."""
    current: list[tuple[int, int]] = []
    k, m = start_i, start_j
    while (
        k < len(overlap_a)
        and m < len(overlap_b)
        and _tokens_match(overlap_a[k], overlap_b[m], max_time_diff)
    ):
        current.append((k, m))
        k += 1
        m += 1
    return current


def merge_longest_contiguous(
    a: list[AlignedToken],
    b: list[AlignedToken],
    *,
    overlap_duration: float,
) -> list[AlignedToken]:
    """Merge two token lists using longest contiguous matching sequence."""
    if not a or not b:
        return b if not a else a

    if a[-1].end <= b[0].start:
        return a + b

    overlap_a, overlap_b, a_end_time, b_start_time = _get_overlap_regions(
        a, b, overlap_duration
    )
    cutoff_time = (a_end_time + b_start_time) / 2

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        return _merge_by_cutoff(a, b, cutoff_time)

    max_time_diff = overlap_duration / 2
    best_contiguous = _find_best_contiguous_match(overlap_a, overlap_b, max_time_diff)
    enough_pairs = len(overlap_a) // 2

    if len(best_contiguous) < enough_pairs:
        raise RuntimeError(f"No pairs exceeding {enough_pairs}")

    a_start_idx = len(a) - len(overlap_a)
    return _merge_with_pairs(a, b, best_contiguous, a_start_idx)


def _compute_lcs_dp_table(
    overlap_a: list[AlignedToken],
    overlap_b: list[AlignedToken],
    max_time_diff: float,
) -> list[list[int]]:
    """Compute the LCS dynamic programming table."""
    dp = [[0 for _ in range(len(overlap_b) + 1)] for _ in range(len(overlap_a) + 1)]

    for i in range(1, len(overlap_a) + 1):
        for j in range(1, len(overlap_b) + 1):
            if _tokens_match(overlap_a[i - 1], overlap_b[j - 1], max_time_diff):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp


def _backtrack_lcs_pairs(
    dp: list[list[int]],
    overlap_a: list[AlignedToken],
    overlap_b: list[AlignedToken],
    max_time_diff: float,
) -> list[tuple[int, int]]:
    """Backtrack through DP table to find LCS pairs."""
    lcs_pairs: list[tuple[int, int]] = []
    i, j = len(overlap_a), len(overlap_b)

    while i > 0 and j > 0:
        if _tokens_match(overlap_a[i - 1], overlap_b[j - 1], max_time_diff):
            lcs_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_pairs.reverse()
    return lcs_pairs


def merge_longest_common_subsequence(
    a: list[AlignedToken],
    b: list[AlignedToken],
    *,
    overlap_duration: float,
) -> list[AlignedToken]:
    """Merge two token lists using longest common subsequence algorithm."""
    if not a or not b:
        return b if not a else a

    if a[-1].end <= b[0].start:
        return a + b

    overlap_a, overlap_b, a_end_time, b_start_time = _get_overlap_regions(
        a, b, overlap_duration
    )
    cutoff_time = (a_end_time + b_start_time) / 2

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        return _merge_by_cutoff(a, b, cutoff_time)

    max_time_diff = overlap_duration / 2
    dp = _compute_lcs_dp_table(overlap_a, overlap_b, max_time_diff)
    lcs_pairs = _backtrack_lcs_pairs(dp, overlap_a, overlap_b, max_time_diff)

    if not lcs_pairs:
        return _merge_by_cutoff(a, b, cutoff_time)

    a_start_idx = len(a) - len(overlap_a)
    return _merge_with_pairs(a, b, lcs_pairs, a_start_idx)
