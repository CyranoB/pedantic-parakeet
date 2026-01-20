"""Tests for backend registry and resolution."""

import pytest

from pedantic_parakeet.backends.base import Backend, STTCapabilities, ModelInfo
from pedantic_parakeet.backends.registry import (
    MODEL_REGISTRY,
    get_model_info,
    list_models,
    resolve_model,
)


class TestModelRegistry:
    """Tests for the MODEL_REGISTRY constant."""

    def test_registry_not_empty(self):
        """Registry should contain curated models."""
        assert len(MODEL_REGISTRY) > 0

    def test_registry_has_parakeet_v3(self):
        """Registry should include the default Parakeet v3 model."""
        assert "mlx-community/parakeet-tdt-0.6b-v3" in MODEL_REGISTRY

    def test_registry_has_whisper(self):
        """Registry should include Whisper model."""
        assert "mlx-community/whisper-large-v3-turbo-asr-fp16" in MODEL_REGISTRY

    def test_registry_has_voxtral(self):
        """Registry should include Voxtral model."""
        assert "mlx-community/Voxtral-Mini-3B-2507-bf16" in MODEL_REGISTRY

    def test_all_entries_are_model_info(self):
        """All registry entries should be ModelInfo instances."""
        for model_id, info in MODEL_REGISTRY.items():
            assert isinstance(info, ModelInfo)
            assert info.model_id == model_id

    def test_all_entries_have_valid_backend(self):
        """All registry entries should have a valid Backend enum."""
        for info in MODEL_REGISTRY.values():
            assert isinstance(info.backend, Backend)

    def test_all_entries_have_capabilities(self):
        """All registry entries should have STTCapabilities."""
        for info in MODEL_REGISTRY.values():
            assert isinstance(info.capabilities, STTCapabilities)


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_returns_model_info_for_known_model(self):
        """Should return ModelInfo for a known model ID."""
        info = get_model_info("mlx-community/parakeet-tdt-0.6b-v3")
        assert info is not None
        assert info.model_id == "mlx-community/parakeet-tdt-0.6b-v3"

    def test_returns_none_for_unknown_model(self):
        """Should return None for unknown model ID."""
        info = get_model_info("unknown/model")
        assert info is None

    def test_returns_none_for_alias(self):
        """Should return None for alias (use resolve_model for alias lookup)."""
        info = get_model_info("parakeet")
        assert info is None  # Aliases require resolve_model


class TestListModels:
    """Tests for list_models function."""

    def test_returns_all_models_by_default(self):
        """Should return all models when no filter specified."""
        models = list_models()
        assert len(models) == len(MODEL_REGISTRY)

    def test_filters_by_parakeet_backend(self):
        """Should filter to Parakeet backend models."""
        models = list_models(backend=Backend.PARAKEET)
        assert len(models) >= 1
        assert all(m.backend == Backend.PARAKEET for m in models)

    def test_filters_by_mlx_audio_backend(self):
        """Should filter to mlx-audio backend models."""
        models = list_models(backend=Backend.MLX_AUDIO)
        assert len(models) >= 1
        assert all(m.backend == Backend.MLX_AUDIO for m in models)


class TestResolveModel:
    """Tests for resolve_model function."""

    def test_resolves_exact_model_id(self):
        """Should resolve exact HuggingFace model ID."""
        info = resolve_model("mlx-community/parakeet-tdt-0.6b-v3")
        assert info.model_id == "mlx-community/parakeet-tdt-0.6b-v3"

    def test_resolves_alias_parakeet(self):
        """Should resolve 'parakeet' alias to v3 model."""
        info = resolve_model("parakeet")
        assert info.model_id == "mlx-community/parakeet-tdt-0.6b-v3"

    def test_resolves_alias_whisper(self):
        """Should resolve 'whisper' alias to Whisper Turbo."""
        info = resolve_model("whisper")
        assert info.model_id == "mlx-community/whisper-large-v3-turbo-asr-fp16"

    def test_resolves_alias_voxtral(self):
        """Should resolve 'voxtral' alias to Voxtral Mini."""
        info = resolve_model("voxtral")
        assert info.model_id == "mlx-community/Voxtral-Mini-3B-2507-bf16"

    def test_raises_value_error_for_unknown_model(self):
        """Should raise ValueError with helpful message for unknown model."""
        with pytest.raises(ValueError) as exc_info:
            resolve_model("unknown/model-id")

        error_msg = str(exc_info.value)
        assert "Unknown model" in error_msg
        assert "unknown/model-id" in error_msg
        assert "Supported models" in error_msg
        assert "Aliases" in error_msg


class TestBackendAssignment:
    """Tests for correct backend assignment in registry."""

    def test_parakeet_v3_uses_parakeet_backend(self):
        """Parakeet v3 should use the Parakeet backend."""
        info = resolve_model("parakeet")
        assert info.backend == Backend.PARAKEET

    def test_parakeet_v2_uses_mlx_audio_backend(self):
        """Parakeet v2 should use the mlx-audio backend."""
        info = resolve_model("parakeet-v2")
        assert info.backend == Backend.MLX_AUDIO

    def test_whisper_uses_mlx_audio_backend(self):
        """Whisper should use the mlx-audio backend."""
        info = resolve_model("whisper")
        assert info.backend == Backend.MLX_AUDIO

    def test_voxtral_uses_mlx_audio_backend(self):
        """Voxtral should use the mlx-audio backend."""
        info = resolve_model("voxtral")
        assert info.backend == Backend.MLX_AUDIO


class TestCapabilities:
    """Tests for capability flags in registry."""

    def test_parakeet_v3_supports_timestamps(self):
        """Parakeet v3 should support timestamps."""
        info = resolve_model("parakeet")
        assert info.capabilities.supports_timestamps is True

    def test_parakeet_v3_supports_language_bias(self):
        """Parakeet v3 should support language bias."""
        info = resolve_model("parakeet")
        assert info.capabilities.supports_language_bias is True

    def test_whisper_supports_timestamps(self):
        """Whisper should support timestamps."""
        info = resolve_model("whisper")
        assert info.capabilities.supports_timestamps is True

    def test_whisper_supports_language_hint(self):
        """Whisper should support language hints."""
        info = resolve_model("whisper")
        assert info.capabilities.supports_language_hint is True

    def test_whisper_does_not_support_language_bias(self):
        """Whisper should not support language bias (Parakeet-specific)."""
        info = resolve_model("whisper")
        assert info.capabilities.supports_language_bias is False

    def test_voxtral_does_not_support_timestamps(self):
        """Voxtral should NOT support timestamps (text-only output)."""
        info = resolve_model("voxtral")
        assert info.capabilities.supports_timestamps is False
