from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class MemorySnapshot:
    active_gb: float
    peak_gb: float


def snapshot_memory() -> MemorySnapshot:
    try:
        return MemorySnapshot(
            active_gb=mx.get_active_memory() / (1024**3),
            peak_gb=mx.get_peak_memory() / (1024**3),
        )
    except Exception:
        return MemorySnapshot(active_gb=0.0, peak_gb=0.0)


def format_memory(s: MemorySnapshot) -> str:
    return f"active={s.active_gb:.2f}GB peak={s.peak_gb:.2f}GB"
