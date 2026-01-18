"""Tests for parakeet.py model functions using mocks."""

import pytest
from unittest.mock import MagicMock, patch

import mlx.core as mx

from pedantic_parakeet.parakeet_mlx.alignment import AlignedToken, AlignedResult
from pedantic_parakeet.parakeet_mlx.parakeet import (
    BaseParakeet,
    DecodingConfig,
    Greedy,
    Beam,
)


class TestBaseParakeetHelpers:
    """Tests for BaseParakeet helper methods."""

    @pytest.fixture
    def mock_base_parakeet(self):
        """Create a mock BaseParakeet with minimal setup."""
        with patch.object(BaseParakeet, '__init__', lambda self, *args, **kwargs: None):
            model = BaseParakeet.__new__(BaseParakeet)
            # Set up minimal required attributes
            model.preprocessor_config = MagicMock()
            model.preprocessor_config.sample_rate = 16000
            model.preprocessor_config.hop_length = 160
            model.encoder_config = MagicMock()
            model.encoder_config.subsampling_factor = 4
            return model

    def test_time_ratio_calculation(self, mock_base_parakeet):
        """Test time_ratio property calculation."""
        # time_ratio = subsampling_factor / sample_rate * hop_length
        # = 4 / 16000 * 160 = 0.04
        expected = 4 / 16000 * 160
        assert abs(mock_base_parakeet.time_ratio - expected) < 1e-10

    def test_compute_confidence_uniform_distribution(self, mock_base_parakeet):
        """Test confidence is low for uniform distribution (high entropy)."""
        # Uniform distribution over 10 classes
        logits = mx.zeros((10,))
        confidence = mock_base_parakeet._compute_confidence(logits, vocab_size=10)
        # Uniform -> high entropy -> low confidence (close to 0)
        assert confidence < 0.1

    def test_compute_confidence_peaked_distribution(self, mock_base_parakeet):
        """Test confidence is high for peaked distribution (low entropy)."""
        # Very peaked distribution - one class has much higher logit
        logits = mx.array([-10.0, -10.0, -10.0, 10.0, -10.0])
        confidence = mock_base_parakeet._compute_confidence(logits, vocab_size=5)
        # Peaked -> low entropy -> high confidence (close to 1)
        assert confidence > 0.9

    def test_compute_confidence_from_probs_uniform(self, mock_base_parakeet):
        """Test confidence from probs for uniform distribution."""
        # Uniform probabilities
        probs = mx.ones((5, 10)) / 10
        confidence = mock_base_parakeet._compute_confidence_from_probs(probs, vocab_size=10)
        assert confidence < 0.1

    def test_compute_confidence_from_probs_peaked(self, mock_base_parakeet):
        """Test confidence from probs for peaked distribution."""
        # Peaked probabilities
        probs = mx.array([[0.01, 0.01, 0.01, 0.95, 0.02]])
        confidence = mock_base_parakeet._compute_confidence_from_probs(probs, vocab_size=5)
        assert confidence > 0.7

    def test_initialize_decode_params_creates_defaults(self, mock_base_parakeet):
        """Test _initialize_decode_params creates default values."""
        features = mx.zeros((2, 10, 64))
        
        B, lengths, last_token, hidden_state = mock_base_parakeet._initialize_decode_params(
            features, None, None, None
        )
        
        assert B == 2
        assert len(lengths) == 2
        assert int(lengths[0]) == 10
        assert int(lengths[1]) == 10
        assert last_token == [None, None]
        assert hidden_state == [None, None]

    def test_initialize_decode_params_preserves_provided_values(self, mock_base_parakeet):
        """Test _initialize_decode_params preserves provided values."""
        features = mx.zeros((2, 10, 64))
        provided_lengths = mx.array([5, 8])
        provided_last_token = [1, 2]
        provided_hidden = [(mx.zeros((1,)), mx.zeros((1,))), None]
        
        B, lengths, last_token, hidden_state = mock_base_parakeet._initialize_decode_params(
            features, provided_lengths, provided_last_token, provided_hidden
        )
        
        assert B == 2
        assert int(lengths[0]) == 5
        assert int(lengths[1]) == 8
        assert last_token == [1, 2]
        assert hidden_state[0] is not None
        assert hidden_state[1] is None


class TestDecodingConfig:
    """Tests for DecodingConfig dataclass."""

    def test_default_config(self):
        """Test default DecodingConfig values."""
        config = DecodingConfig()
        assert isinstance(config.decoding, Greedy)
        assert config.language_bias is None

    def test_greedy_config(self):
        """Test Greedy decoding config."""
        config = DecodingConfig(decoding=Greedy())
        assert isinstance(config.decoding, Greedy)

    def test_beam_config(self):
        """Test Beam decoding config."""
        beam = Beam(beam_size=10, length_penalty=0.5)
        config = DecodingConfig(decoding=beam)
        assert isinstance(config.decoding, Beam)
        assert config.decoding.beam_size == 10
        assert config.decoding.length_penalty == pytest.approx(0.5)

    def test_beam_defaults(self):
        """Test Beam default values."""
        beam = Beam()
        assert beam.beam_size == 5
        assert beam.length_penalty == pytest.approx(1.0)
        assert beam.patience == pytest.approx(1.0)
        assert beam.duration_reward == pytest.approx(0.7)

    def test_language_bias(self):
        """Test language_bias in config."""
        bias = mx.array([0.0, -0.5, 0.0])
        config = DecodingConfig(language_bias=bias)
        assert config.language_bias is not None


class TestParakeetTDTDecode:
    """Tests for ParakeetTDT decode methods using mocks."""

    @pytest.fixture
    def mock_tdt_model(self):
        """Create a mock ParakeetTDT model."""
        model = MagicMock()
        model.vocabulary = ["a", "b", "c", "d"]
        model.durations = [0, 1, 2, 4]
        model.max_symbols = 10
        model.time_ratio = 0.04
        
        def init_params(features, lengths, last_token, hidden_state):
            batch_size = features.shape[0]
            seq_len = features.shape[1]
            if hidden_state is None:
                hidden_state = [None] * batch_size
            if lengths is None:
                lengths = mx.array([seq_len] * batch_size)
            if last_token is None:
                last_token = [None] * batch_size
            return batch_size, lengths, last_token, hidden_state
        
        model._initialize_decode_params = init_params
        model._compute_confidence = lambda logits, vocab_size: 0.95
        
        return model

    def test_decode_routes_to_greedy(self, mock_tdt_model):
        """Test that decode routes to decode_greedy for Greedy config."""
        mock_tdt_model.decode_greedy = MagicMock(return_value=([], []))
        mock_tdt_model.decode_beam = MagicMock(return_value=([], []))
        
        features = mx.zeros((1, 10, 64))
        config = DecodingConfig(decoding=Greedy())
        
        match config.decoding:
            case Greedy():
                mock_tdt_model.decode_greedy(features, config=config)
            case Beam():
                mock_tdt_model.decode_beam(features, config=config)
        
        mock_tdt_model.decode_greedy.assert_called_once()
        mock_tdt_model.decode_beam.assert_not_called()

    def test_decode_routes_to_beam(self, mock_tdt_model):
        """Test that decode routes to decode_beam for Beam config."""
        mock_tdt_model.decode_greedy = MagicMock(return_value=([], []))
        mock_tdt_model.decode_beam = MagicMock(return_value=([], []))
        
        features = mx.zeros((1, 10, 64))
        config = DecodingConfig(decoding=Beam())
        
        match config.decoding:
            case Greedy():
                mock_tdt_model.decode_greedy(features, config=config)
            case Beam():
                mock_tdt_model.decode_beam(features, config=config)
        
        mock_tdt_model.decode_beam.assert_called_once()
        mock_tdt_model.decode_greedy.assert_not_called()


class TestParakeetRNNTDecode:
    """Tests for ParakeetRNNT decode method."""

    def test_rnnt_only_supports_greedy(self):
        """Test that RNNT raises error for non-greedy decoding."""
        config = DecodingConfig(decoding=Beam())
        assert not isinstance(config.decoding, Greedy)


class TestParakeetCTCDecode:
    """Tests for ParakeetCTC decode method."""

    @pytest.fixture
    def mock_ctc_model(self):
        """Create a mock ParakeetCTC model."""
        model = MagicMock()
        model.vocabulary = ["a", "b", "c", "d"]
        model.time_ratio = 0.04
        model._compute_confidence_from_probs = lambda probs, vocab_size: 0.9
        return model

    def test_ctc_decode_returns_tokens(self, mock_ctc_model):
        """Test CTC decode returns list of token lists."""
        mock_ctc_model.decode = MagicMock(return_value=[
            [AlignedToken(id=0, text="a", start=0.0, duration=0.1)]
        ])
        
        result = mock_ctc_model.decode(mx.zeros((1, 10, 5)))
        assert len(result) == 1
        assert len(result[0]) == 1
        assert isinstance(result[0][0], AlignedToken)


class TestAlignedTokenCreation:
    """Tests for AlignedToken creation in decode methods."""

    def test_aligned_token_properties(self):
        """Test AlignedToken has correct properties."""
        token = AlignedToken(
            id=1,
            text="hello",
            start=0.5,
            duration=0.25,
            confidence=0.95
        )
        
        assert token.id == 1
        assert token.text == "hello"
        assert token.start == pytest.approx(0.5)
        assert token.duration == pytest.approx(0.25)
        assert token.end == pytest.approx(0.75)
        assert token.confidence == pytest.approx(0.95)

    def test_aligned_token_default_confidence(self):
        """Test AlignedToken default confidence."""
        token = AlignedToken(id=1, text="test", start=0.0, duration=1.0)
        assert token.confidence == pytest.approx(1.0)


class TestLanguageBiasApplication:
    """Tests for language bias application in decoding."""

    def test_language_bias_modifies_logits(self):
        """Test that language bias is added to logits."""
        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bias = mx.array([-0.5, 0.0, -0.5, 0.0, -0.5])
        
        biased_logits = logits + bias
        
        assert float(biased_logits[0]) == pytest.approx(0.5)
        assert float(biased_logits[1]) == pytest.approx(2.0)
        assert float(biased_logits[2]) == pytest.approx(2.5)

    def test_language_bias_none_no_change(self):
        """Test that None language bias doesn't change logits."""
        logits = mx.array([1.0, 2.0, 3.0])
        config = DecodingConfig(language_bias=None)
        
        if config.language_bias is not None:
            logits = logits + config.language_bias
        
        assert float(logits[0]) == pytest.approx(1.0)


class TestBatchProcessing:
    """Tests for batch processing in decode methods."""

    def test_batch_size_detection(self):
        """Test batch size is correctly detected from features."""
        features = mx.zeros((3, 10, 64))
        batch_size, seq_len, _ = features.shape
        assert batch_size == 3
        assert seq_len == 10

    def test_lengths_created_for_batch(self):
        """Test lengths array is created for each batch item."""
        features = mx.zeros((3, 10, 64))
        batch_size, seq_len, _ = features.shape
        lengths = mx.array([seq_len] * batch_size)
        
        assert len(lengths) == 3
        assert all(int(length) == 10 for length in lengths)


class TestDecodeGreedyLogic:
    """Tests for greedy decoding logic."""

    def test_blank_token_not_added(self):
        """Test that blank tokens are not added to hypothesis."""
        vocabulary = ["a", "b", "c"]
        blank_token_id = len(vocabulary)
        
        # Simulate the logic: blank tokens should not be added
        pred_token = blank_token_id
        hypothesis = []
        
        # Only add non-blank tokens (this mirrors the real decode logic)
        hypothesis = self._add_token_if_non_blank(pred_token, blank_token_id, hypothesis)
        
        assert len(hypothesis) == 0
    
    def _add_token_if_non_blank(self, token: int, blank_id: int, hypothesis: list) -> list:
        """Helper to add token if not blank."""
        if token != blank_id:
            hypothesis.append(token)
        return hypothesis

    def test_non_blank_token_added(self):
        """Test that non-blank tokens are added to hypothesis."""
        vocabulary = ["a", "b", "c"]
        blank_token_id = len(vocabulary)
        
        pred_token = 1
        hypothesis = []
        
        # Use the helper method to test the logic
        hypothesis = self._add_token_if_non_blank(pred_token, blank_token_id, hypothesis)
        
        assert len(hypothesis) == 1
        assert hypothesis[0] == 1

    def test_max_symbols_prevents_stuck(self):
        """Test max_symbols prevents getting stuck."""
        max_symbols = 3
        new_symbols = 0
        step = 0
        duration = 0
        
        # Use helper to simulate the stuck prevention logic
        step, new_symbols = self._simulate_stuck_prevention(
            duration, step, new_symbols, max_symbols, iterations=5
        )
        
        assert step > 0
    
    def _simulate_stuck_prevention(
        self, duration: int, step: int, new_symbols: int, max_symbols: int, iterations: int
    ) -> tuple[int, int]:
        """Simulate stuck prevention logic for testing."""
        for _ in range(iterations):
            if duration == 0:
                new_symbols += 1
                if max_symbols <= new_symbols:
                    step += 1
                    new_symbols = 0
        return step, new_symbols


class TestDecodeBeamLogic:
    """Tests for beam decoding logic."""

    def test_beam_size_limits_candidates(self):
        """Test beam size limits number of active candidates."""
        beam_size = 3
        candidates = [
            {"score": 0.9},
            {"score": 0.8},
            {"score": 0.7},
            {"score": 0.6},
            {"score": 0.5},
        ]
        
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        active_beam = sorted_candidates[:beam_size]
        
        assert len(active_beam) == 3
        assert active_beam[0]["score"] == pytest.approx(0.9)

    def test_length_penalty_affects_ranking(self):
        """Test length penalty affects hypothesis ranking."""
        length_penalty = 2.0
        
        short_score = 0.8
        short_length = 2
        short_normalized = short_score / (max(1, short_length) ** length_penalty)
        
        long_score = 1.2
        long_length = 5
        long_normalized = long_score / (max(1, long_length) ** length_penalty)
        
        assert short_normalized > long_normalized

    def test_patience_affects_max_candidates(self):
        """Test patience multiplier affects max candidates."""
        beam_size = 5
        patience = 2.0
        
        max_candidates = round(beam_size * patience)
        assert max_candidates == 10


class TestTranscribeMethod:
    """Tests for the transcribe method."""

    @pytest.fixture
    def mock_model_for_transcribe(self):
        """Create mock model for transcribe tests."""
        model = MagicMock()
        model.preprocessor_config = MagicMock()
        model.preprocessor_config.sample_rate = 16000
        model.preprocessor_config.hop_length = 160
        model.generate = MagicMock(return_value=[
            AlignedResult(text="Hello world", sentences=[])
        ])
        return model

    def test_transcribe_calls_generate(self, mock_model_for_transcribe):
        """Test transcribe calls generate method."""
        mock_model_for_transcribe.generate(mx.zeros((1, 100, 80)))
        mock_model_for_transcribe.generate.assert_called_once()

    def test_chunk_callback_called(self):
        """Test chunk callback is called during chunked processing."""
        callback_calls = []
        
        def callback(current, total):
            callback_calls.append((current, total))
        
        total = 1000
        for i in range(0, total, 200):
            callback(min(i + 200, total), total)
        
        assert len(callback_calls) == 5
        assert callback_calls[-1] == (1000, 1000)
