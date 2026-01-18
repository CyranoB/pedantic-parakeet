import math
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from . import tokenizer
from .alignment import (
    AlignedResult,
    AlignedToken,
    SentenceConfig,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)
from .audio import PreprocessArgs, get_logmel, load_audio
from .cache import ConformerCache, RotatingConformerCache
from .conformer import Conformer, ConformerArgs
from .ctc import AuxCTCArgs, ConvASRDecoder, ConvASRDecoderArgs
from .rnnt import JointArgs, JointNetwork, PredictArgs, PredictNetwork


@dataclass
class TDTDecodingArgs:
    model_type: str
    durations: list[int]
    greedy: dict | None


@dataclass
class RNNTDecodingArgs:
    greedy: dict | None


@dataclass
class CTCDecodingArgs:
    greedy: dict | None


@dataclass
class ParakeetTDTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: TDTDecodingArgs


@dataclass
class ParakeetRNNTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: RNNTDecodingArgs


@dataclass
class ParakeetCTCArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: ConvASRDecoderArgs
    decoding: CTCDecodingArgs


@dataclass
class ParakeetTDTCTCArgs(ParakeetTDTArgs):
    aux_ctc: AuxCTCArgs


# API
@dataclass
class Greedy:
    pass


@dataclass
class Beam:
    beam_size: int = 5
    length_penalty: float = 1.0
    patience: float = 1.0
    duration_reward: float = 0.7  # TDT-only


@dataclass
class DecodingConfig:
    decoding: Union[Greedy, Beam] = field(default_factory=Greedy)
    sentence: SentenceConfig = field(default_factory=SentenceConfig)
    language_bias: mx.array | None = None


# common methods
class BaseParakeet(nn.Module):
    """Base parakeet model for interface purpose"""

    def __init__(self, preprocess_args: PreprocessArgs, encoder_args: ConformerArgs):
        super().__init__()

        self.preprocessor_config = preprocess_args
        self.encoder_config = encoder_args

        self.encoder = Conformer(encoder_args)

    @property
    def time_ratio(self) -> float:
        return (
            self.encoder_config.subsampling_factor
            / self.preprocessor_config.sample_rate
            * self.preprocessor_config.hop_length
        )

    def _compute_confidence(
        self, token_logits: mx.array, vocab_size: int
    ) -> float:
        """Compute confidence score using entropy-based method."""
        token_probs = mx.softmax(token_logits, axis=-1)
        entropy = -mx.sum(token_probs * mx.log(token_probs + 1e-10), axis=-1)
        max_entropy = mx.log(mx.array(vocab_size, dtype=token_probs.dtype))
        return float(1.0 - (entropy / max_entropy))

    def _compute_confidence_from_probs(
        self, token_probs: mx.array, vocab_size: int
    ) -> float:
        """Compute confidence score from probabilities using entropy-based method."""
        entropies = -mx.sum(token_probs * mx.log(token_probs + 1e-10), axis=-1)
        avg_entropy = mx.mean(entropies)
        max_entropy = mx.log(mx.array(vocab_size, dtype=token_probs.dtype))
        return float(1.0 - (avg_entropy / max_entropy))

    def _initialize_decode_params(
        self,
        features: mx.array,
        lengths: Optional[mx.array],
        last_token: Optional[list[Optional[int]]],
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]],
    ) -> tuple[int, mx.array, list[Optional[int]], list[Optional[tuple[mx.array, mx.array]]]]:
        """Initialize parameters for decoding."""
        B, S, *_ = features.shape

        if hidden_state is None:
            hidden_state = list([None] * B)

        if lengths is None:
            lengths = mx.array([S] * B)

        if last_token is None:
            last_token = list([None] * B)

        return B, lengths, last_token, hidden_state

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        """
        Generate transcription results from the Parakeet model, handling batches and single input.
        Args:
            mel (mx.array):
                Mel-spectrogram input with shape [batch, sequence, mel_dim] for
                batch processing or [sequence, mel_dim] for single input.
            decoding_config (DecodingConfig, optional):
                Configuration object that controls decoding behavior and
                parameters for the generation process. Defaults to DecodingConfig().
        Returns:
            list[AlignedResult]: List of transcription results with aligned tokens
                and sentences, one for each input in the batch.
        """
        raise NotImplementedError

    def _transcribe_without_chunking(
        self,
        audio_data: mx.array,
        decoding_config: DecodingConfig,
    ) -> AlignedResult:
        """Transcribe audio data in a single pass without chunking."""
        mel = get_logmel(audio_data, self.preprocessor_config)
        return self.generate(mel, decoding_config=decoding_config)[0]

    def _offset_chunk_tokens(
        self,
        chunk_result: AlignedResult,
        chunk_offset: float,
    ) -> None:
        """Apply time offset to all tokens in a chunk result."""
        for sentence in chunk_result.sentences:
            for token in sentence.tokens:
                token.start += chunk_offset
                token.end = token.start + token.duration

    def _merge_chunk_tokens(
        self,
        all_tokens: list[AlignedToken],
        new_tokens: list[AlignedToken],
        overlap_duration: float,
    ) -> list[AlignedToken]:
        """Merge new tokens with existing tokens, handling overlap."""
        if not all_tokens:
            return new_tokens

        try:
            return merge_longest_contiguous(
                all_tokens, new_tokens, overlap_duration=overlap_duration
            )
        except RuntimeError:
            return merge_longest_common_subsequence(
                all_tokens, new_tokens, overlap_duration=overlap_duration
            )

    def _transcribe_chunked(
        self,
        audio_data: mx.array,
        decoding_config: DecodingConfig,
        chunk_duration: float,
        overlap_duration: float,
        chunk_callback: Optional[Callable],
    ) -> AlignedResult:
        """Transcribe audio data in chunks with overlap."""
        sample_rate = self.preprocessor_config.sample_rate
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        step_size = chunk_samples - overlap_samples

        all_tokens: list[AlignedToken] = []

        for start in range(0, len(audio_data), step_size):
            end = min(start + chunk_samples, len(audio_data))

            if chunk_callback is not None:
                chunk_callback(end, len(audio_data))

            if end - start < self.preprocessor_config.hop_length:
                break  # prevent zero-length log mel

            chunk_audio = audio_data[start:end]
            chunk_mel = get_logmel(chunk_audio, self.preprocessor_config)
            chunk_result = self.generate(chunk_mel, decoding_config=decoding_config)[0]

            chunk_offset = start / sample_rate
            self._offset_chunk_tokens(chunk_result, chunk_offset)

            all_tokens = self._merge_chunk_tokens(
                all_tokens, chunk_result.tokens, overlap_duration
            )

        return sentences_to_result(
            tokens_to_sentences(all_tokens, decoding_config.sentence)
        )

    def transcribe(
        self,
        path: Path | str,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        decoding_config: DecodingConfig = DecodingConfig(),
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 15.0,
        chunk_callback: Optional[Callable] = None,
    ) -> AlignedResult:
        """
        Transcribe an audio file, with optional chunking for long files.
        Args:
            path (Path | str):
                Path to the audio file to be transcribed.
            dtype (mx.Dtype, optional):
                Data type for processing the audio. Defaults to mx.bfloat16.
            chunk_duration (float, optional):
                If provided, splits audio into chunks of this length (in seconds)
                for processing. When None, processes the entire file at once.
                Defaults to None.
            overlap_duration (float, optional):
                Overlap between consecutive chunks in seconds. Only used when
                chunk_duration is specified. Defaults to 15.0.
            chunk_callback (Callable, optional):
                A function to call when each chunk is processed. The callback
                is called with (current_position, total_position) arguments
                to track progress. Defaults to None.
        Returns:
            AlignedResult: Transcription result with aligned tokens and sentences.
        """
        audio_path = Path(path)
        audio_data = load_audio(audio_path, self.preprocessor_config.sample_rate)

        # Check if chunking is needed
        audio_length_seconds = len(audio_data) / self.preprocessor_config.sample_rate
        needs_chunking = chunk_duration is not None and audio_length_seconds > chunk_duration

        if not needs_chunking:
            return self._transcribe_without_chunking(audio_data, decoding_config)

        # chunk_duration is guaranteed to be not None here due to the check above
        return self._transcribe_chunked(
            audio_data, decoding_config, typing.cast(float, chunk_duration), overlap_duration, chunk_callback
        )

    def transcribe_stream(
        self,
        context_size: tuple[int, int] = (256, 256),
        depth=1,
        *,
        keep_original_attention: bool = False,
        decoding_config: DecodingConfig = DecodingConfig(),
    ) -> "StreamingParakeet":
        """
        Create a StreamingParakeet object for real-time (streaming) inference.
        Args:
            context_size (tuple[int, int], optional):
                A pair (left_context, right_context) for attention context windows.
            depth (int, optional):
                How many encoder layers will carry over their key/value
                cache (i.e. hidden state) exactly across chunks. Because
                we use local (non-causal) attention, the cache is only
                guaranteed to match a full forward pass up through each
                cached layer:
                    • depth=1 (default): only the first encoder layer's
                    cache matches exactly.
                    • depth=2: the first two layers match, and so on.
                    • depth=N (model's total layers): full equivalence to
                    a non-streaming forward pass.
                Setting `depth` larger than the model's total number
                of encoder layers won't have any impacts.
            keep_original_attention (bool, optional):
                Whether to preserve the original attention class
                during streaming inference. Defaults to False. (Will switch to local attention.)
            decoding_config (DecodingConfig, optional):
                Configuration object that controls decoding behavior
                Defaults to DecodingConfig().
        Returns:
            StreamingParakeet: A context manager for streaming inference.
        """
        return StreamingParakeet(
            self,
            context_size,
            depth,
            decoding_config=decoding_config,
            keep_original_attention=keep_original_attention,
        )


# models
class ParakeetTDT(BaseParakeet):
    """MLX Implementation of Parakeet-TDT Model"""

    def __init__(self, args: ParakeetTDTArgs):
        super().__init__(args.preprocessor, args.encoder)

        assert args.decoding.model_type == "tdt", "Model must be a TDT model"

        self.vocabulary = args.joint.vocabulary
        self.durations = args.decoding.durations
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def _run_decoder_joint_pass(
        self,
        feature: mx.array,
        step: int,
        token_input: Optional[int],
        decoder_hidden_state: Optional[tuple[mx.array, mx.array]],
    ) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array]]:
        """Run decoder and joint network for a single step."""
        decoder_out, (hidden, cell) = self.decoder(
            mx.array([[token_input]]) if token_input is not None else None,
            decoder_hidden_state,
        )
        decoder_out = decoder_out.astype(feature.dtype)
        decoder_hidden = (
            hidden.astype(feature.dtype),
            cell.astype(feature.dtype),
        )
        joint_out = self.joint(feature[:, step : step + 1], decoder_out)
        return joint_out, decoder_out, decoder_hidden

    def _apply_language_bias(
        self,
        token_logits: mx.array,
        language_bias: Optional[mx.array],
    ) -> mx.array:
        """Apply language bias to token logits if provided."""
        if language_bias is not None:
            return token_logits + language_bias
        return token_logits

    def _handle_stuck_prevention(
        self,
        duration: int,
        current_step: int,
        new_symbols: int,
    ) -> tuple[int, int]:
        """Handle stuck prevention logic for TDT decoding."""
        if duration != 0:
            return current_step + duration, 0

        new_symbols += 1
        if self.max_symbols is not None and self.max_symbols <= new_symbols:
            return current_step + 1, 0

        return current_step + duration, new_symbols

    def decode(
        self,
        features: mx.array,
        lengths: Optional[mx.array] = None,
        last_token: Optional[list[Optional[int]]] = None,
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[Optional[tuple[mx.array, mx.array]]]]:
        """Run TDT decoder with features, optional length and decoder state. Outputs list[list[AlignedToken]] and updated hidden state"""
        mx.eval(features)

        match config.decoding:
            case Greedy():
                return self.decode_greedy(
                    features, lengths, last_token, hidden_state, config=config
                )
            case Beam():
                return self.decode_beam(
                    features, lengths, last_token, hidden_state, config=config
                )
            case _:
                raise NotImplementedError(
                    f"{config.decoding} is not supported in TDT models."
                )

    def _get_top_k_indices(
        self,
        logprobs: mx.array,
        k: int,
    ) -> List[int]:
        """Get top-k indices from log probabilities."""
        return typing.cast(
            List[int],
            mx.argpartition(logprobs, -k)[-k:].tolist(),
        )

    def _compute_beam_step_update(
        self,
        duration: int,
        current_step: int,
        current_stuck: int,
    ) -> tuple[int, int]:
        """Compute next step and stuck count for beam search."""
        stuck = 0 if duration != 0 else current_stuck + 1

        if self.max_symbols is not None and stuck >= self.max_symbols:
            return current_step + 1, 0

        return current_step + duration, stuck

    def _compute_hypothesis_score(
        self,
        base_score: float,
        token_logprob: float,
        duration_logprob: float,
        duration_reward: float,
    ) -> float:
        """Compute combined score for a hypothesis."""
        return (
            base_score
            + token_logprob * (1 - duration_reward)
            + duration_logprob * duration_reward
        )

    def _merge_hypothesis_scores(
        self,
        candidates: Dict[int, Any],
        key: int,
        new_hypothesis: Any,
    ) -> None:
        """Merge hypothesis with same path using log-sum-exp."""
        if key not in candidates:
            candidates[key] = new_hypothesis
            return

        other = candidates[key]
        maxima = max(other.score, new_hypothesis.score)
        merged_score = maxima + math.log(
            math.exp(other.score - maxima) + math.exp(new_hypothesis.score - maxima)
        )

        if new_hypothesis.score > other.score:
            candidates[key] = new_hypothesis
        candidates[key].score = merged_score

    def _select_best_hypothesis(
        self,
        hypotheses: list,
        length_penalty: float,
    ) -> Any:
        """Select best hypothesis with length penalty normalization."""
        return max(
            hypotheses,
            key=lambda x: x.score / (max(1, len(x.tokens)) ** length_penalty),
        )

    def _create_beam_token(
        self,
        token_id: int,
        step: int,
        duration: int,
        token_logprob: float,
        duration_logprob: float,
    ) -> AlignedToken:
        """Create an aligned token for beam search."""
        return AlignedToken(
            id=token_id,
            start=step * self.time_ratio,
            duration=duration * self.time_ratio,
            confidence=math.exp(token_logprob + duration_logprob),
            text=tokenizer.decode([token_id], self.vocabulary),
        )

    def _expand_hypothesis(
        self,
        hyp: Any,
        token_id: int,
        duration_idx: int,
        token_logprob: float,
        duration_logprob: float,
        decoder_hidden: tuple[mx.array, mx.array],
        duration_reward: float,
        blank_token_id: int,
    ) -> Any:
        """Expand a hypothesis with a new token/duration decision."""
        from dataclasses import dataclass as inner_dataclass
        
        dur = self.durations[duration_idx]
        is_blank = token_id == blank_token_id
        step, stuck = self._compute_beam_step_update(dur, hyp.step, hyp.stuck)
        score = self._compute_hypothesis_score(
            hyp.score, token_logprob, duration_logprob, duration_reward
        )

        new_tokens = hyp.tokens
        if not is_blank:
            new_token = self._create_beam_token(
                token_id, hyp.step, dur, token_logprob, duration_logprob
            )
            new_tokens = list(hyp.tokens) + [new_token]

        # Return dict-like structure since we can't access the Hypothesis class here
        return {
            "score": score,
            "step": step,
            "last_token": hyp.last_token if is_blank else token_id,
            "hidden_state": hyp.hidden_state if is_blank else decoder_hidden,
            "stuck": stuck,
            "tokens": new_tokens,
        }

    def _process_beam_hypothesis(
        self,
        hyp: Any,
        feature: mx.array,
        config: DecodingConfig,
        beam_token: int,
        beam_duration: int,
        blank_token_id: int,
    ) -> list[dict]:
        """Process a single hypothesis and generate all expansions."""
        joint_out, _, decoder_hidden = self._run_decoder_joint_pass(
            feature, hyp.step, hyp.last_token, hyp.hidden_state
        )

        token_logits = joint_out[0, 0, 0, : blank_token_id + 1]
        token_logits = self._apply_language_bias(token_logits, config.language_bias)
        duration_logits = joint_out[0, 0, 0, blank_token_id + 1 :]

        token_logprobs = nn.log_softmax(token_logits, -1)
        duration_logprobs = nn.log_softmax(duration_logits, -1)

        token_k = self._get_top_k_indices(token_logprobs, beam_token)
        duration_k = self._get_top_k_indices(duration_logprobs, beam_duration)

        token_logprobs_list = typing.cast(List[float], token_logprobs.tolist())
        duration_logprobs_list = typing.cast(List[float], duration_logprobs.tolist())

        expansions = []
        for tok in token_k:
            for dec in duration_k:
                expansion = self._expand_hypothesis(
                    hyp, tok, dec,
                    token_logprobs_list[tok], duration_logprobs_list[dec],
                    decoder_hidden, config.decoding.duration_reward, blank_token_id
                )
                expansions.append(expansion)

        return expansions

    def decode_beam(
        self,
        features: mx.array,
        lengths: Optional[mx.array] = None,
        last_token: Optional[list[Optional[int]]] = None,
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[Optional[tuple[mx.array, mx.array]]]]:
        assert isinstance(config.decoding, Beam)

        beam_config = config.decoding
        beam_token = min(beam_config.beam_size, len(self.vocabulary) + 1)
        beam_duration = min(beam_config.beam_size, len(self.durations))
        max_candidates = round(beam_config.beam_size * beam_config.patience)
        blank_token_id = len(self.vocabulary)

        @dataclass
        class Hypothesis:
            score: float
            step: int
            last_token: Optional[int]
            hidden_state: Optional[tuple[mx.array, mx.array]]
            stuck: int
            tokens: list[AlignedToken]

            def __hash__(self) -> int:
                return hash((self.step, tuple((x.id for x in self.tokens))))

        B, lengths, last_token, hidden_state = self._initialize_decode_params(
            features, lengths, last_token, hidden_state
        )

        results = []
        results_hidden = []
        for batch in range(B):
            feature = features[batch : batch + 1]
            length = int(lengths[batch])

            finished: list[Hypothesis] = []
            active_beam: list[Hypothesis] = [
                Hypothesis(0.0, 0, last_token[batch], hidden_state[batch], 0, [])
            ]

            while len(finished) < max_candidates and active_beam:
                candidates: Dict[int, Hypothesis] = {}

                for hyp in active_beam:
                    expansions = self._process_beam_hypothesis(
                        hyp, feature, config, beam_token, beam_duration, blank_token_id
                    )
                    for exp in expansions:
                        new_hyp = Hypothesis(**exp)
                        self._merge_hypothesis_scores(candidates, hash(new_hyp), new_hyp)

                finished.extend([h for h in candidates.values() if h.step >= length])
                active_beam = sorted(
                    [h for h in candidates.values() if h.step < length],
                    key=lambda x: x.score, reverse=True
                )[: beam_config.beam_size]

            all_hypotheses = finished + active_beam
            if not all_hypotheses:
                results.append([])
                results_hidden.append(hidden_state[batch])
            else:
                best = self._select_best_hypothesis(all_hypotheses, beam_config.length_penalty)
                results.append(best.tokens)
                results_hidden.append(best.hidden_state)

        return results, results_hidden

    def _decode_greedy_single_batch(
        self,
        feature: mx.array,
        length: int,
        batch_last_token: Optional[int],
        batch_hidden_state: Optional[tuple[mx.array, mx.array]],
        config: DecodingConfig,
    ) -> tuple[list[AlignedToken], Optional[int], Optional[tuple[mx.array, mx.array]]]:
        """Decode a single batch using greedy TDT decoding."""
        hypothesis: list[AlignedToken] = []
        step = 0
        new_symbols = 0
        blank_token_id = len(self.vocabulary)

        while step < length:
            joint_out, _, decoder_hidden = self._run_decoder_joint_pass(
                feature, step, batch_last_token, batch_hidden_state
            )

            token_logits = joint_out[0, 0, :, : blank_token_id + 1]
            token_logits = self._apply_language_bias(token_logits, config.language_bias)

            pred_token = int(mx.argmax(token_logits))
            confidence = self._compute_confidence(token_logits, blank_token_id + 1)
            decision = int(mx.argmax(joint_out[0, 0, :, blank_token_id + 1 :]))
            duration = self.durations[decision]

            if pred_token != blank_token_id:
                hypothesis.append(
                    AlignedToken(
                        int(pred_token),
                        start=step * self.time_ratio,
                        duration=duration * self.time_ratio,
                        confidence=confidence,
                        text=tokenizer.decode([pred_token], self.vocabulary),
                    )
                )
                batch_last_token = pred_token
                batch_hidden_state = decoder_hidden

            step, new_symbols = self._handle_stuck_prevention(duration, step, new_symbols)

        return hypothesis, batch_last_token, batch_hidden_state

    def decode_greedy(
        self,
        features: mx.array,
        lengths: Optional[mx.array] = None,
        last_token: Optional[list[Optional[int]]] = None,
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[Optional[tuple[mx.array, mx.array]]]]:
        assert isinstance(config.decoding, Greedy)  # type guarntee

        B, lengths, last_token, hidden_state = self._initialize_decode_params(
            features, lengths, last_token, hidden_state
        )

        results = []
        for batch in range(B):
            feature = features[batch : batch + 1]
            length = int(lengths[batch])

            hypothesis, last_token[batch], hidden_state[batch] = self._decode_greedy_single_batch(
                feature, length, last_token[batch], hidden_state[batch], config
            )
            results.append(hypothesis)

        return results, hidden_state

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)

        result, _ = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(
                tokens_to_sentences(hypothesis, decoding_config.sentence)
            )
            for hypothesis in result
        ]


class ParakeetRNNT(BaseParakeet):
    """MLX Implementation of Parakeet-RNNT Model"""

    def __init__(self, args: ParakeetRNNTArgs):
        super().__init__(args.preprocessor, args.encoder)

        self.vocabulary = args.joint.vocabulary
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(
        self,
        features: mx.array,
        lengths: Optional[mx.array] = None,
        last_token: Optional[list[Optional[int]]] = None,
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[Optional[tuple[mx.array, mx.array]]]]:
        """Run TDT decoder with features, optional length and decoder state. Outputs list[list[AlignedToken]] and updated hidden state"""
        assert isinstance(config.decoding, Greedy), (
            "Only greedy decoding is supported for RNNT decoder now"
        )

        B, lengths, last_token, hidden_state = self._initialize_decode_params(
            features, lengths, last_token, hidden_state
        )

        results = []
        for batch in range(B):
            hypothesis = []

            feature = features[batch : batch + 1]
            length = int(lengths[batch])

            step = 0
            new_symbols = 0

            while step < length:
                # decoder pass
                decoder_out, (hidden, cell) = self.decoder(
                    mx.array([[last_token[batch]]])
                    if last_token[batch] is not None
                    else None,
                    hidden_state[batch],
                )
                decoder_out = decoder_out.astype(feature.dtype)
                decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                # joint pass
                joint_out = self.joint(feature[:, step : step + 1], decoder_out)

                # sampling
                token_logits = joint_out[0, 0]
                pred_token = int(mx.argmax(token_logits))

                # compute confidence score
                vocab_size = len(self.vocabulary) + 1
                confidence = self._compute_confidence(token_logits, vocab_size)

                # rnnt decoding rule
                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=step * self.time_ratio,
                            duration=1 * self.time_ratio,
                            confidence=confidence,
                            text=tokenizer.decode([pred_token], self.vocabulary),
                        )
                    )
                    last_token[batch] = pred_token
                    hidden_state[batch] = decoder_hidden

                    # prevent stucking
                    new_symbols += 1
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        step += 1
                        new_symbols = 0
                else:
                    step += 1
                    new_symbols = 0

            results.append(hypothesis)

        return results, hidden_state

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)

        result, _ = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(
                tokens_to_sentences(hypothesis, decoding_config.sentence)
            )
            for hypothesis in result
        ]


class ParakeetCTC(BaseParakeet):
    """MLX Implementation of Parakeet-CTC Model"""

    def __init__(self, args: ParakeetCTCArgs):
        super().__init__(args.preprocessor, args.encoder)

        self.vocabulary = args.decoder.vocabulary

        self.decoder = ConvASRDecoder(args.decoder)

    def _create_ctc_token(
        self,
        token_id: int,
        start_frame: int,
        end_frame: int,
        probs: mx.array,
    ) -> AlignedToken:
        """Create an aligned token with computed timing and confidence."""
        vocab_size = len(self.vocabulary) + 1
        token_probs = probs[start_frame:end_frame]
        confidence = self._compute_confidence_from_probs(token_probs, vocab_size)

        return AlignedToken(
            token_id,
            start=start_frame * self.time_ratio,
            duration=(end_frame - start_frame) * self.time_ratio,
            confidence=confidence,
            text=tokenizer.decode([token_id], self.vocabulary),
        )

    def _find_last_non_blank_frame(
        self,
        best_tokens: mx.array,
        start_frame: int,
        length: int,
    ) -> int:
        """Find the last non-blank frame from the end."""
        blank_token_id = len(self.vocabulary)
        for t in range(length - 1, start_frame, -1):
            if int(best_tokens[t]) != blank_token_id:
                return t
        return length - 1

    def _decode_ctc_single_batch(
        self,
        predictions: mx.array,
        length: int,
    ) -> list[AlignedToken]:
        """Decode a single batch using CTC decoding."""
        best_tokens = mx.argmax(predictions, axis=1)
        probs = mx.exp(predictions)
        blank_token_id = len(self.vocabulary)

        hypothesis: list[AlignedToken] = []
        token_boundaries: list[tuple[int, None]] = []
        prev_token = -1

        for t, token_id in enumerate(best_tokens):
            token_idx = int(token_id)

            # Skip blank tokens and repeated tokens
            if token_idx == blank_token_id or token_idx == prev_token:
                continue

            # Emit previous token if exists
            if prev_token != -1:
                start_frame = token_boundaries[-1][0]
                hypothesis.append(
                    self._create_ctc_token(prev_token, start_frame, t, probs)
                )

            token_boundaries.append((t, None))
            prev_token = token_idx

        # Handle final token
        if prev_token != -1:
            start_frame = token_boundaries[-1][0]
            last_non_blank = self._find_last_non_blank_frame(best_tokens, start_frame, length)
            hypothesis.append(
                self._create_ctc_token(prev_token, start_frame, last_non_blank + 1, probs)
            )

        return hypothesis

    def decode(
        self,
        features: mx.array,
        lengths: mx.array,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> list[list[AlignedToken]]:
        """Run CTC decoder with features and lengths. Outputs list[list[AlignedToken]]."""
        B, _, *_ = features.shape

        logits = self.decoder(features)
        mx.eval(logits, lengths)

        results = []
        for batch in range(B):
            length = int(lengths[batch])
            predictions = logits[batch, :length]
            hypothesis = self._decode_ctc_single_batch(predictions, length)
            results.append(hypothesis)

        return results

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)

        result = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(
                tokens_to_sentences(hypothesis, decoding_config.sentence)
            )
            for hypothesis in result
        ]


class ParakeetTDTCTC(ParakeetTDT):
    """MLX Implementation of Parakeet-TDT-CTC Model

    Has ConvASRDecoder decoder in `.ctc_decoder` but `.generate` uses TDT decoder all the times (Please open an issue if you need CTC decoder use-case!)"""

    def __init__(self, args: ParakeetTDTCTCArgs):
        super().__init__(args)

        self.ctc_decoder = ConvASRDecoder(args.aux_ctc.decoder)


# streaming
class StreamingParakeet:
    model: "BaseParakeet"
    cache: List[ConformerCache]

    audio_buffer: mx.array
    mel_buffer: Optional[mx.array]
    decoder_hidden: Optional[tuple[mx.array, mx.array]] = None
    last_token: Optional[int] = None

    finalized_tokens: list[AlignedToken]
    draft_tokens: list[AlignedToken]

    context_size: tuple[int, int]
    depth: int
    decoding_config: DecodingConfig
    keep_original_attention: bool = False

    def __init__(
        self,
        model: "BaseParakeet",
        context_size: tuple[int, int],
        depth: int = 1,
        *,
        keep_original_attention: bool = False,
        decoding_config: DecodingConfig = DecodingConfig(),
    ) -> None:
        self.context_size = context_size
        self.depth = depth
        self.decoding_config = decoding_config
        self.keep_original_attention = keep_original_attention

        self.model = model
        self.cache = [
            RotatingConformerCache(self.keep_size, cache_drop_size=self.drop_size)
            for _ in range(len(model.encoder.layers))
        ]

        self.audio_buffer = mx.array([])
        self.mel_buffer = None
        self.finalized_tokens = []
        self.draft_tokens = []

    def __enter__(self):
        if not self.keep_original_attention:
            self.model.encoder.set_attention_model(
                "rel_pos_local_attn", self.context_size
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.keep_original_attention:
            self.model.encoder.set_attention_model(
                "rel_pos"
            )  # hard-coded; might cache if there's actually new varient than rel_pos
        del self.audio_buffer
        del self.cache

        mx.clear_cache()

    @property
    def keep_size(self):
        """Indicates how many encoded feature frames to keep in KV cache"""
        return self.context_size[0]

    @property
    def drop_size(self):
        """Indicates how many encoded feature frames to drop"""
        return self.context_size[1] * self.depth

    @property
    def result(self) -> AlignedResult:
        """Transcription result"""
        return sentences_to_result(
            tokens_to_sentences(
                self.finalized_tokens + self.draft_tokens, self.decoding_config.sentence
            )
        )

    def add_audio(self, audio: mx.array) -> None:
        """Takes portion of audio and transcribe it.

        `audio` must be 1D array"""

        self.audio_buffer = mx.concat(
            [
                self.audio_buffer,
                audio,
            ],
            axis=0,
        )
        mel = get_logmel(
            self.audio_buffer[
                : (
                    len(self.audio_buffer)
                    // self.model.preprocessor_config.hop_length
                    * self.model.preprocessor_config.hop_length
                )
            ],
            self.model.preprocessor_config,
        )

        if self.mel_buffer is None:  # init
            self.mel_buffer = mel
        else:
            self.mel_buffer = mx.concat([self.mel_buffer, mel], axis=1)

        self.audio_buffer = self.audio_buffer[
            (mel.shape[1] * self.model.preprocessor_config.hop_length) :
        ]

        features, lengths = self.model.encoder(
            self.mel_buffer[
                :,
                : (
                    self.mel_buffer.shape[1]
                    // self.model.encoder_config.subsampling_factor
                    * self.model.encoder_config.subsampling_factor
                ),
            ],
            cache=self.cache,
        )
        mx.eval(features, lengths)
        length = int(lengths[0])

        # cache will automatically dropped in cache level
        leftover = self.mel_buffer.shape[1] - (
            length * self.model.encoder_config.subsampling_factor
        )
        self.mel_buffer = self.mel_buffer[
            :,
            -(
                self.drop_size * self.model.encoder_config.subsampling_factor + leftover
            ) :,
        ]

        # we decode in two phase
        # first phase: finalized region decode
        # second phase: draft region decode (will be dropped)
        finalized_length = max(0, length - self.drop_size)

        if isinstance(self.model, ParakeetTDT) or isinstance(self.model, ParakeetRNNT):
            finalized_tokens, finalized_state = self.model.decode(
                features,
                mx.array([finalized_length]),
                [self.last_token],
                [self.decoder_hidden],
                config=self.decoding_config,
            )

            self.decoder_hidden = finalized_state[0]
            self.last_token = (
                finalized_tokens[0][-1].id if len(finalized_tokens[0]) > 0 else None
            )

            draft_tokens, _ = self.model.decode(
                features[:, finalized_length:],
                mx.array(
                    [
                        features[:, finalized_length:].shape[1]
                    ]  # i believe in lazy evaluation
                ),
                [self.last_token],
                [self.decoder_hidden],
                config=self.decoding_config,
            )

            self.finalized_tokens.extend(finalized_tokens[0])
            self.draft_tokens = draft_tokens[0]
        elif isinstance(self.model, ParakeetCTC):
            finalized_tokens = self.model.decode(
                features, mx.array([finalized_length]), config=self.decoding_config
            )

            draft_tokens = self.model.decode(
                features[:, finalized_length:],
                mx.array(
                    [
                        features[:, finalized_length:].shape[1]
                    ]  # i believe in lazy evaluation
                ),
                config=self.decoding_config,
            )

            self.finalized_tokens.extend(finalized_tokens[0])
            self.draft_tokens = draft_tokens[0]
        else:
            raise NotImplementedError("This model does not support real-time decoding")
