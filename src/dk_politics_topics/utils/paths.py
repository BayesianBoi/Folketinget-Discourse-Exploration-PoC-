from pathlib import Path
from typing import Optional


def repo_root(current: Optional[Path] = None) -> Path:
    """Resolve repository root based on file location."""
    if current is None:
        current = Path(__file__).resolve()
    return current.parents[3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_path(filename: str, base_dir: Path) -> Path:
    return base_dir / filename
