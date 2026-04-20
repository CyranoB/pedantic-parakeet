---
status: resolved
trigger: "ImportError: cannot import name 'load' from 'mlx_audio.stt'"
created: 2026-01-20T03:35:22Z
updated: 2026-01-20T03:48:00Z
---

## Resolution
root_cause: mlx-audio v0.2.10 moved the model loading logic. 'load' function is not exported in 'mlx_audio.stt', but 'load_model' exists in 'mlx_audio.stt.utils'.
fix: Updated `pedantic_parakeet/backends/mlx_audio.py` to import `load_model` from `mlx_audio.stt.utils` instead of `load` from `mlx_audio.stt`.
verification: Verified that `pedantic-parakeet` no longer crashes with ImportError when using the mlx-audio backend.
files_changed: 
- pedantic_parakeet/backends/mlx_audio.py
