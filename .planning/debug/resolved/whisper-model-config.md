---
status: resolved
trigger: "ModelDimensions.__init__() got an unexpected keyword argument 'activation_dropout'"
created: 2026-01-20T03:50:00Z
updated: 2026-01-20T04:10:00Z
---

## Resolution
root_cause: |
  mlx-audio 0.2.10 has two bugs in its Whisper model loading:
  1. The `from_pretrained` method passes all HuggingFace config keys directly to 
     `ModelDimensions`, but HuggingFace configs have many extra fields (activation_dropout,
     activation_function, etc.) that ModelDimensions doesn't accept.
  2. It looks for `weights.safetensors` but HuggingFace models use `model.safetensors`.
  
  Additionally, the config key names differ between HuggingFace format and what
  mlx-audio expects (e.g., `num_mel_bins` vs `n_mels`).

fix: |
  Implemented `_load_whisper_model()` function in `pedantic_parakeet/backends/mlx_audio.py`
  that:
  1. Manually maps HuggingFace config keys to ModelDimensions fields
  2. Tries multiple weight file names (weights.safetensors, model.safetensors, weights.npz)
  
  The `_load_model()` method now uses this custom loader for Whisper models.

verification: |
  Successfully transcribed a 30-second French audio clip:
  ```
  uv run pedantic-parakeet recording_30s.wav -f txt -v -l fr -m mlx-community/whisper-large-v3-turbo-asr-fp16
  Loading model: mlx-community/whisper-large-v3-turbo-asr-fp16...
    ✓ recording_30s.txt
  ✓ 1 file(s) transcribed
  ```
  All 147 tests pass.

files_changed: 
  - pedantic_parakeet/backends/mlx_audio.py
