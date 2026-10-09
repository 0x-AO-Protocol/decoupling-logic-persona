from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ao_da.pipeline.signal_collector import SanmeiProfile, TurnInput, VitalSnapshot, collect_turn_input


@dataclass(frozen=True)
class WhatGold:
    required_actions: tuple[str, ...]
    required_concepts: tuple[str, ...]
    forbidden: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentTask:
    id: str
    description: str
    profile: SanmeiProfile
    vitals: VitalSnapshot
    user_prompt: str
    what_gold: WhatGold

    def to_turn_input(self) -> TurnInput:
        return collect_turn_input(
            profile=self.profile,
            vitals=self.vitals,
            user_prompt=self.user_prompt,
        )


def _parse_what_gold(raw: dict[str, Any]) -> WhatGold:
    return WhatGold(
        required_actions=tuple(raw.get("required_actions") or ()),
        required_concepts=tuple(raw.get("required_concepts") or ()),
        forbidden=tuple(raw.get("forbidden") or ()),
    )


def load_alignment_tasks(
    path: Path | None = None,
    *,
    task_ids: list[str] | None = None,
) -> list[AlignmentTask]:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "config" / "alignment_tax_tasks.yaml"
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    tasks: list[AlignmentTask] = []
    for row in doc.get("tasks") or []:
        tid = row["id"]
        if task_ids and tid not in task_ids:
            continue
        tasks.append(
            AlignmentTask(
                id=tid,
                description=str(row.get("description") or ""),
                profile=SanmeiProfile(**row["profile"]),
                vitals=VitalSnapshot(**row["vitals"]),
                user_prompt=str(row["user_prompt"]).strip(),
                what_gold=_parse_what_gold(row["what_gold"]),
            )
        )
    if task_ids:
        missing = set(task_ids) - {t.id for t in tasks}
        if missing:
            raise KeyError(f"Unknown alignment task ids: {sorted(missing)}")
    return tasks
