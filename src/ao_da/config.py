from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@dataclass(frozen=True)
class AssetPaths:
    gemma_training_root: Path
    repo_root: Path
    bases: dict[str, Path]
    adapters: dict[str, Path]

    def base(self, key: str) -> Path:
        return self.bases[key]

    def adapter(self, key: str) -> Path:
        return self.adapters[key]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / value).resolve()


def load_asset_paths(config_path: Path | None = None) -> AssetPaths:
    """Load model/adapter paths from config/paths.local.yaml."""
    if config_path is None:
        config_path = CONFIG_DIR / "paths.local.yaml"
        if not config_path.exists():
            config_path = CONFIG_DIR / "paths.local.yaml.example"

    with config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    gemma_root = Path(raw["gemma_training_root"]).expanduser().resolve()
    repo_root = Path(raw.get("repo_root", REPO_ROOT)).expanduser().resolve()

    if not gemma_root.is_dir():
        raise FileNotFoundError(f"gemma_training_root not found: {gemma_root}")

    bases = {k: _resolve(gemma_root, v) for k, v in raw["bases"].items()}

    adapters: dict[str, Path] = {}
    for key, value in raw["adapters"].items():
        if isinstance(value, str) and value.startswith("repo:"):
            adapters[key] = _resolve(repo_root, value.removeprefix("repo:"))
        else:
            adapters[key] = _resolve(gemma_root, str(value))

    return AssetPaths(
        gemma_training_root=gemma_root,
        repo_root=repo_root,
        bases=bases,
        adapters=adapters,
    )


def validate_assets(paths: AssetPaths) -> list[str]:
    """Return list of missing paths (empty = all required assets present)."""
    missing: list[str] = []
    for name, path in {**paths.bases, **paths.adapters}.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    return missing
