import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .logging import get_logger

logger = get_logger(__name__)


def hash_config(config: Dict[str, Any]) -> str:
    """Create a short, deterministic hash from a config dict."""
    serialized = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(serialized).hexdigest()[:12]


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_numpy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    logger.info("Saved numpy cache to %s", path)


def load_numpy(path: Path, mmap_mode: Optional[str] = None) -> np.ndarray:
    logger.info("Loading numpy cache from %s (mmap_mode=%s)", path, mmap_mode)
    return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)


def cached_path(base_dir: Path, name: str, ext: str) -> Path:
    return base_dir / f"{name}.{ext}"


def maybe_load_numpy(base_dir: Path, name: str, mmap_mode: Optional[str] = None) -> Optional[np.ndarray]:
    path = cached_path(base_dir, name, "npy")
    if path.exists():
        return load_numpy(path, mmap_mode=mmap_mode)
    return None
