"""Curated model registry for STT models.

Provides a registry of known models with their backend assignments
and capability metadata. Used by CLI for model listing and by
transcriber for backend selection.
"""

from .base import Backend, ModelInfo, STTCapabilities

# Curated registry of known STT models
MODEL_REGISTRY: dict[str, ModelInfo] = {
    # Parakeet TDT v3 - best quality, supports language bias
    "mlx-community/parakeet-tdt-0.6b-v3": ModelInfo(
        model_id="mlx-community/parakeet-tdt-0.6b-v3",
        backend=Backend.PARAKEET,
        capabilities=STTCapabilities(
            supports_timestamps=True,
            supports_language_bias=True,
            supports_language_hint=False,
            supports_chunking=True,
        ),
        description="Parakeet TDT 0.6B v3 - High accuracy, language bias support",
        aliases=["parakeet-v3", "parakeet"],
    ),
    # Parakeet TDT v2 - older model via mlx-audio
    "mlx-community/parakeet-tdt-0.6b-v2": ModelInfo(
        model_id="mlx-community/parakeet-tdt-0.6b-v2",
        backend=Backend.MLX_AUDIO,
        capabilities=STTCapabilities(
            supports_timestamps=True,
            supports_language_bias=False,
            supports_language_hint=False,
            supports_chunking=True,
        ),
        description="Parakeet TDT 0.6B v2 - Previous version via mlx-audio",
        aliases=["parakeet-v2"],
    ),
    # Whisper Large v3 Turbo - fast, multilingual
    "mlx-community/whisper-large-v3-turbo-asr-fp16": ModelInfo(
        model_id="mlx-community/whisper-large-v3-turbo-asr-fp16",
        backend=Backend.MLX_AUDIO,
        capabilities=STTCapabilities(
            supports_timestamps=True,
            supports_language_bias=False,
            supports_language_hint=True,
            supports_chunking=False,
        ),
        description="Whisper Large v3 Turbo - Fast, 100+ languages",
        aliases=["whisper-turbo", "whisper"],
    ),
    # Voxtral Mini - LLM-based transcription
    "mlx-community/Voxtral-Mini-3B-2507-bf16": ModelInfo(
        model_id="mlx-community/Voxtral-Mini-3B-2507-bf16",
        backend=Backend.MLX_AUDIO,
        capabilities=STTCapabilities(
            supports_timestamps=False,
            supports_language_bias=False,
            supports_language_hint=True,
            supports_chunking=False,
        ),
        description="Voxtral Mini 3B - LLM-based, context-aware transcription",
        aliases=["voxtral", "voxtral-mini"],
    ),
}


def get_model_info(model_id: str) -> ModelInfo | None:
    """Get model info by exact model ID.

    Args:
        model_id: Full HuggingFace model identifier.

    Returns:
        ModelInfo if found, None otherwise.
    """
    return MODEL_REGISTRY.get(model_id)


def list_models(backend: Backend | None = None) -> list[ModelInfo]:
    """List all curated models, optionally filtered by backend.

    Args:
        backend: Filter to specific backend, or None for all.

    Returns:
        List of ModelInfo for matching models.
    """
    models = list(MODEL_REGISTRY.values())
    if backend is not None:
        models = [m for m in models if m.backend == backend]
    return models


def resolve_model(model_id: str) -> ModelInfo:
    """Resolve a model ID or alias to ModelInfo.

    Args:
        model_id: Full model ID or short alias.

    Returns:
        ModelInfo for the resolved model.

    Raises:
        ValueError: If model ID is not found in registry.
    """
    # Try exact match first
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]

    # Try alias lookup
    for info in MODEL_REGISTRY.values():
        if model_id in info.aliases:
            return info

    # Not found - provide helpful error
    supported = sorted(MODEL_REGISTRY.keys())
    aliases = sorted(
        alias for info in MODEL_REGISTRY.values() for alias in info.aliases
    )

    raise ValueError(
        f"Unknown model: '{model_id}'. "
        f"Supported models: {supported}. "
        f"Aliases: {aliases}."
    )
