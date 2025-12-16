from __future__ import annotations

import time
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import EmbeddingConfig, PathsConfig
from ..utils.cache import cached_path, hash_config, maybe_load_numpy, save_numpy
from ..utils.logging import get_logger

logger = get_logger(__name__)

try:  # torch is required by sentence-transformers; guard for clarity
    import torch
except Exception:  # pragma: no cover
    torch = None


def _resolve_device(preferred: str) -> str:
    """Pick a usable device: prefer user-specified, else mps, cuda, cpu."""
    if preferred and preferred.lower() != "auto":
        return preferred
    if torch is not None:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def embed_texts(
    texts: Iterable[str],
    paths: PathsConfig,
    cfg: EmbeddingConfig,
) -> np.ndarray:
    """Compute sentence-transformer embeddings with optional caching."""
    texts_list = list(texts)
    cache_name = cfg.cache_name or hash_config({"model": cfg.model_name, "n": len(texts_list)})

    if cfg.cache_embeddings:
        cached = maybe_load_numpy(paths.embeddings_dir, cache_name)
        if cached is not None:
            return cached

    device = _resolve_device(cfg.device)
    model = None
    errors = []
    for candidate in [cfg.model_name, cfg.model_fallback]:
        if not candidate:
            continue
        try:
            logger.info("Loading embedding model: %s on device=%s", candidate, device)
            model = SentenceTransformer(candidate, device=device)
            break
        except Exception as exc:  # pragma: no cover - defensive
            errors.append((candidate, str(exc)))
            logger.warning("Failed to load %s (%s); trying next if available", candidate, exc)

    if model is None:
        raise RuntimeError(f"Could not load any embedding model. Errors: {errors}")

    logger.info("Encoding %d texts (batch_size=%d, device=%s)", len(texts_list), cfg.batch_size, device)
    start = time.time()
    embeddings = model.encode(
        texts_list,
        batch_size=cfg.batch_size,
        show_progress_bar=cfg.show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logger.info("Finished encoding in %.1f seconds; shape=%s", time.time() - start, embeddings.shape)

    if cfg.cache_embeddings:
        save_numpy(embeddings, cached_path(paths.embeddings_dir, cache_name, "npy"))

    return embeddings
