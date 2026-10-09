from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten


def adapter_weights_file(adapter_dir: str | Path) -> Path | None:
    """Resolve MLX LoRA safetensors file inside an adapter directory."""
    adapter_dir = Path(adapter_dir)
    for name in ("adapters.safetensors", "adapter.safetensors"):
        candidate = adapter_dir / name
        if candidate.exists():
            return candidate
    return None


def load_flat_lora(adapter_dir: str | Path) -> dict[str, mx.array]:
    """Load LoRA weights as a flat dict (compatible with gemma-training api_server)."""
    path = adapter_weights_file(adapter_dir)
    if path is None:
        raise FileNotFoundError(f"No adapter safetensors in {adapter_dir}")
    return dict(mx.load(str(path)))


def unflatten_lora(flat: dict[str, mx.array]):
    return tree_unflatten(list(flat.items()))


def adapter_size_bytes(adapter_dir: str | Path) -> int:
    path = adapter_weights_file(adapter_dir)
    return path.stat().st_size if path else 0
