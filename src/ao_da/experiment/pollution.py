from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from ao_da.experiment.tasks import AlignmentTask

WhatPathMode = Literal["clean", "polluted"]
HowPathMode = Literal["clean", "polluted"]

_CONFIG_CACHE: dict[str, Any] | None = None


def _config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "alignment_tax_pollution.yaml"


def load_pollution_config() -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with _config_path().open(encoding="utf-8") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def list_pollution_levels() -> list[str]:
    cfg = load_pollution_config()
    return list((cfg.get("levels") or {}).keys())


def _repeat_to_length(text: str, target_chars: int) -> str:
    if target_chars <= 0:
        return ""
    text = text.strip()
    if not text:
        return ""
    chunks: list[str] = []
    total = 0
    n = 0
    while total < target_chars:
        chunk = f"[block {n + 1}] {text}\n"
        chunks.append(chunk)
        total += len(chunk)
        n += 1
    return "".join(chunks)[:target_chars]


def _fixtures_root() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "fixtures"


def _load_history_fixture(fixture_key: str) -> list[dict[str, str]]:
    path = _fixtures_root() / "chat_history" / f"{fixture_key}.yaml"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        rows = doc.get("turns") or []
        return [{"role": str(r["role"]), "content": str(r["content"]).strip()} for r in rows]

    cfg = load_pollution_config()
    rows = (cfg.get("history_fixtures") or {}).get(fixture_key) or []
    return [{"role": str(r["role"]), "content": str(r["content"]).strip()} for r in rows]


def _load_system_bloat(persona: str, key: str) -> str:
    if key == "by_persona":
        path = _fixtures_root() / "system_bloat" / f"{persona.lower()}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        cfg = load_pollution_config()
        return str((cfg.get("system_bloat_fixtures") or {}).get(persona.lower(), "")).strip()
    path = _fixtures_root() / "system_bloat" / f"{key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    cfg = load_pollution_config()
    return str((cfg.get("system_bloat_fixtures") or {}).get(key, "")).strip()


def _resolve_history(cfg: dict[str, Any], level_spec: dict[str, Any], task: AlignmentTask) -> list[dict[str, str]]:
    key = level_spec.get("history_key")
    if not key:
        return []
    domain_map = cfg.get("task_domain_map") or {}
    domain = domain_map.get(task.id, "sleep_performance")
    fixture_key = domain if key == "by_domain" else str(key)
    return _load_history_fixture(fixture_key)


def _resolve_system_bloat(
    cfg: dict[str, Any], level_spec: dict[str, Any], persona: str
) -> str:
    key = level_spec.get("system_bloat_key")
    if not key:
        return ""
    return _load_system_bloat(persona, str(key))


@dataclass(frozen=True)
class PollutionContext:
    """Reproducible contamination package for message assembly."""

    level: str
    core_user_prompt: str
    effective_user_prompt: str
    chat_history: tuple[dict[str, str], ...]
    system_bloat: str
    what_path_mode: WhatPathMode
    how_path_mode: HowPathMode

    @property
    def history_list(self) -> list[dict[str, str]]:
        return list(self.chat_history)

    def log_fields(self) -> dict[str, Any]:
        return {
            "pollution_level": self.level,
            "what_path_mode": self.what_path_mode,
            "how_path_mode": self.how_path_mode,
            "core_user_chars": len(self.core_user_prompt),
            "effective_user_chars": len(self.effective_user_prompt),
            "history_turns": len(self.chat_history),
            "system_bloat_chars": len(self.system_bloat),
        }


def build_pollution_context(
    task: AlignmentTask,
    level: str,
    persona: str,
    *,
    what_path_mode: WhatPathMode = "clean",
    how_path_mode: HowPathMode = "polluted",
) -> PollutionContext:
    cfg = load_pollution_config()
    levels = cfg.get("levels") or {}
    if level not in levels:
        raise KeyError(f"Unknown pollution level '{level}'. Available: {sorted(levels)}")

    spec = levels[level]
    core = task.user_prompt.strip()
    filler_key = int(spec.get("long_user_chars") or 0)
    template = str(cfg.get("long_user_template") or "")
    long_block = _repeat_to_length(template, filler_key) if filler_key > 0 else ""
    effective = core if not long_block else f"{core}\n\n{long_block}"

    history = _resolve_history(cfg, spec, task)
    bloat = _resolve_system_bloat(cfg, spec, persona)

    # L0 forces clean paths unless explicitly overridden by arm
    if level == "L0":
        w_mode: WhatPathMode = "clean"
        h_mode: HowPathMode = "clean"
    else:
        w_mode = what_path_mode
        h_mode = how_path_mode

    return PollutionContext(
        level=level,
        core_user_prompt=core,
        effective_user_prompt=effective,
        chat_history=tuple(history),
        system_bloat=bloat,
        what_path_mode=w_mode,
        how_path_mode=h_mode,
    )


def estimate_prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(prompt)
        return len(ids)
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return max(1, len(text) // 4)


@dataclass(frozen=True)
class Run1bArm:
    id: str
    description: str
    what_path: WhatPathMode
    how_path: HowPathMode
    group: str  # G0 | G3 | G3_what_polluted


def load_run_1b_arms() -> list[Run1bArm]:
    cfg = load_pollution_config()
    raw = (cfg.get("run_1b") or {}).get("arms") or []
    out: list[Run1bArm] = []
    for row in raw:
        arm_id = row["id"]
        group = "G0" if arm_id == "G0" else "G3"
        if arm_id == "G3_what_polluted":
            group = "G3_what_polluted"
        default_what = "polluted" if arm_id == "G0" else "clean"
        out.append(
            Run1bArm(
                id=arm_id,
                description=str(row.get("description") or ""),
                what_path=row.get("what_path", default_what),
                how_path=row.get("how_path", "polluted"),
                group=group,
            )
        )
    return out


def load_run_1b_defaults() -> dict[str, Any]:
    return load_pollution_config().get("run_1b") or {}
