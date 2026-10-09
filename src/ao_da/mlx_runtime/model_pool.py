from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import mlx.core as mx
from mlx_lm import load

from ao_da.config import AssetPaths
from ao_da.mlx_runtime.lora import adapter_weights_file, load_flat_lora, unflatten_lora
from ao_da.mlx_runtime.memory import MemorySnapshot, snapshot_memory


BaseKey = Literal["llama_8b_4bit", "gemma_4b_4bit"]
AdapterRole = Literal["what", "persona"]


class PersonaId(str, Enum):
    GOKU = "goku"
    MAKIMA = "makima"
    TRINITY = "trinity"


@dataclass
class SwapMetrics:
    adapter_key: str
    resident_ms: float
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot


@dataclass
class ModelPool:
    """Single shared base + hot-swappable What / Persona LoRA adapters (Order 0)."""

    paths: AssetPaths
    base_key: BaseKey = "llama_8b_4bit"
    model: object = field(init=False, repr=False)
    tokenizer: object = field(init=False, repr=False)
    _flat_adapters: dict[str, dict] = field(init=False, default_factory=dict)
    _active_adapter: str | None = field(init=False, default=None)
    _swap_history: list[SwapMetrics] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._persona_map = self._build_persona_map()
        self._what_key = self._what_adapter_key()
        self.mount_initial()

    def _build_persona_map(self) -> dict[PersonaId, str]:
        if self.base_key == "llama_8b_4bit":
            return {
                PersonaId.GOKU: "goku_llama",
                PersonaId.MAKIMA: "makima_llama",
                PersonaId.TRINITY: "trinity_llama",
            }
        return {
            PersonaId.GOKU: "goku_gemma",
            PersonaId.MAKIMA: "makima_gemma",
        }

    def _what_adapter_key(self) -> str:
        return (
            "mental_coach_llama"
            if self.base_key == "llama_8b_4bit"
            else "mental_coach_gemma"
        )

    def _default_persona_path(self) -> str:
        return str(self.paths.adapter(self._persona_map[PersonaId.GOKU]))

    def mount_initial(self) -> None:
        """Load base once; preload flat LoRA tensors for resident hot-swap."""
        base_path = str(self.paths.base(self.base_key))
        self.model, self.tokenizer = load(
            base_path, adapter_path=self._default_persona_path()
        )
        mx.eval(self.model.parameters())

        keys_to_preload = {self._what_key, *self._persona_map.values()}
        for key in keys_to_preload:
            adapter_path = self.paths.adapter(key)
            if adapter_weights_file(adapter_path) is not None:
                self._flat_adapters[key] = load_flat_lora(adapter_path)

        persona_keys = [
            key for key in self._persona_map.values() if key in self._flat_adapters
        ]
        if not persona_keys:
            raise RuntimeError(
                "No persona LoRA adapters loaded. Check gemma-training paths in config/paths.local.yaml."
            )

        self.swap_adapter(persona_keys[0], record=False)

    def swap_adapter(self, adapter_key: str, *, record: bool = True) -> SwapMetrics | None:
        """Atomic adapter swap on the shared base (resident tensors only)."""
        if adapter_key not in self._flat_adapters:
            raise KeyError(
                f"Adapter '{adapter_key}' not preloaded. "
                f"Available: {list(self._flat_adapters)}"
            )
        if self._active_adapter == adapter_key:
            return None

        before = snapshot_memory()
        t0 = time.perf_counter()
        self.model.update(unflatten_lora(self._flat_adapters[adapter_key]))
        mx.eval(self.model.parameters())
        elapsed_ms = (time.perf_counter() - t0) * 1000
        after = snapshot_memory()
        self._active_adapter = adapter_key

        metrics = SwapMetrics(
            adapter_key=adapter_key,
            resident_ms=elapsed_ms,
            memory_before=before,
            memory_after=after,
        )
        if record:
            self._swap_history.append(metrics)
        return metrics

    def resolve_persona_adapter(self, persona: PersonaId) -> tuple[str, str | None]:
        """Return (adapter_key, fallback_note). Trinity uses Makima weights if LoRA missing."""
        key = self._persona_map[persona]
        if key in self._flat_adapters:
            return key, None
        if persona == PersonaId.TRINITY:
            makima_key = self._persona_map[PersonaId.MAKIMA]
            if makima_key in self._flat_adapters:
                return makima_key, "makima_lora_trinity_prompt_overlay"
        raise KeyError(
            f"Persona '{persona.value}' adapter '{key}' not loaded. "
            f"Available: {list(self._flat_adapters)}"
        )

    def swap_persona(self, persona: PersonaId) -> SwapMetrics | None:
        adapter_key, _ = self.resolve_persona_adapter(persona)
        return self.swap_adapter(adapter_key)

    def swap_what(self) -> SwapMetrics | None:
        return self.swap_adapter(self._what_key)

    def stress_swap_cycle(
        self, cycles: int = 5
    ) -> list[tuple[str, float, MemorySnapshot]]:
        """What ↔ Goku ↔ Makima resident swap loop (§6 memory turn prep)."""
        sequence: list[str] = [
            self._what_key,
            self._persona_map[PersonaId.GOKU],
            self._persona_map[PersonaId.MAKIMA],
        ]
        results: list[tuple[str, float, MemorySnapshot]] = []
        for i in range(cycles):
            key = sequence[i % len(sequence)]
            if key not in self._flat_adapters:
                continue
            m = self.swap_adapter(key)
            if m:
                results.append((key, m.resident_ms, m.memory_after))
        return results

    def release(self) -> None:
        del self.model, self.tokenizer
        self.model = None  # type: ignore[assignment]
        self.tokenizer = None  # type: ignore[assignment]
        self._flat_adapters.clear()
        gc.collect()
        try:
            mx.clear_cache()
        except AttributeError:
            pass
