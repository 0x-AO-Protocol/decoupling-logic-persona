from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ArchitectureGroup(str, Enum):
    G0 = "G0"
    G3 = "G3"
    G4 = "G4"


@dataclass(frozen=True)
class RunConfig:
    """Single experiment run identity for §5 logging fields."""

    group: ArchitectureGroup
    persona: str
    base: str
    constrained_decoding: bool
    seed: int
    task_id: str
    pollution_level: str = "L0"
    arm_id: str = ""
    what_path_mode: str = "clean"
    how_path_mode: str = "clean"

    @property
    def log_fields(self) -> dict[str, str | bool | int]:
        return {
            "architecture": self.group.value,
            "persona": self.persona,
            "base_model": self.base,
            "cd_enabled": self.constrained_decoding,
            "seed": self.seed,
            "task_id": self.task_id,
            "pollution_level": self.pollution_level,
            "arm_id": self.arm_id,
            "what_path_mode": self.what_path_mode,
            "how_path_mode": self.how_path_mode,
        }


def group_from_name(name: str) -> ArchitectureGroup:
    return ArchitectureGroup(name.upper())
