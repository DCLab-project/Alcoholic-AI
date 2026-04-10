from __future__ import annotations

import os
from pathlib import Path


def _is_project_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "src").is_dir()
        and (path / "assets").is_dir()
        and (path / "configs").is_dir()
    )


def _iter_candidate_roots(anchor: Path | None = None):
    env_root = os.environ.get("AI_FRIDGE_PROJECT_ROOT")
    if env_root:
        yield Path(env_root).expanduser()

    if anchor is not None:
        resolved_anchor = anchor.expanduser().resolve()
        if resolved_anchor.is_dir():
            yield resolved_anchor
        for parent in resolved_anchor.parents:
            yield parent

    cwd = Path.cwd().resolve()
    yield cwd
    for parent in cwd.parents:
        yield parent

    home = Path.home()
    for name in ("ai_fridge", "ai-fridge-system"):
        yield home / name


def find_project_root(anchor: Path | None = None) -> Path:
    seen = set()
    for candidate in _iter_candidate_roots(anchor):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _is_project_root(candidate):
            return candidate
    raise RuntimeError(
        "Could not locate the ai-fridge-system project root. "
        "Set AI_FRIDGE_PROJECT_ROOT to the repository path if needed."
    )


PROJECT_ROOT = find_project_root(Path(__file__))
ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

INGREDIENT_LABELS_FILE = ASSETS_DIR / "labels" / "ingredient_labels_stable19.txt"
INGREDIENT_CHECKPOINT_DIR = CHECKPOINTS_DIR / "ingredient"
INGREDIENT_DATASET_DIR = PROCESSED_DATA_DIR / "ingredient_cls"


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def to_project_relative(path_like: str | Path) -> str:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
