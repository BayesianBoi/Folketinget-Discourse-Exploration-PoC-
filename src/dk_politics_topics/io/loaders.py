from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from ..config import CorpusConfig, PathsConfig
from ..utils.logging import get_logger
from .schemas import CorpusSchema, validate_corpus_df

logger = get_logger(__name__)

try:
    import pyreadr  # type: ignore
except Exception:  # pragma: no cover - optional dependency for RDS
    pyreadr = None


def _deterministic_doc_id(text: str, prefix: str, suffix: str = "") -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    suffix_part = f"_{suffix}" if suffix else ""
    return f"{prefix}_{digest}{suffix_part}"


def _normalize_party(party: str, mapping: dict) -> str:
    if not isinstance(party, str):
        return "UNKNOWN"
    normalized = party.strip().upper()
    if normalized in mapping:
        return mapping[normalized]
    return normalized


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".rds":
        if pyreadr is None:
            raise ImportError("pyreadr is required to read .rds files")
        df = list(pyreadr.read_r(path).values())[0]
    elif path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
    elif path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    logger.info("Loaded raw corpus with %d rows and columns %s", len(df), df.columns.tolist())
    return df


def normalize_corpus(df: pd.DataFrame, corpus_cfg: CorpusConfig) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={k: v for k, v in corpus_cfg.column_mapping.items() if k in df.columns}, inplace=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "year" not in df.columns and "date" in df.columns:
        df["year"] = df["date"].dt.year

    if "party" in df.columns:
        df["party"] = df["party"].apply(lambda x: _normalize_party(x, corpus_cfg.party_normalization))
    else:
        df["party"] = "UNKNOWN"

    if "text" in df.columns:
        df["text"] = df["text"].fillna("").astype(str)
    else:
        df["text"] = ""
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    if "doc_id" not in df.columns:
        df["doc_id"] = [
            _deterministic_doc_id(text, prefix=corpus_cfg.doc_id_prefix, suffix=str(idx))
            for idx, text in enumerate(df["text"])
        ]

    if "year" not in df.columns and corpus_cfg.start_year:
        df["year"] = corpus_cfg.start_year

    df = df[df["text"].str.len() >= corpus_cfg.min_text_length]

    if "year" in df.columns:
        df = df[(df["year"] >= corpus_cfg.start_year) & (df["year"] <= corpus_cfg.end_year)]

    if corpus_cfg.sample_size:
        df = df.sample(n=min(corpus_cfg.sample_size, len(df)), random_state=42)

    logger.info("Normalized corpus to %d rows", len(df))
    return df


def load_and_validate(
    paths: PathsConfig,
    corpus_cfg: CorpusConfig,
    schema: CorpusSchema | None = None,
    filename: str | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    paths.ensure_dirs()
    filename = filename or "Corpus_speeches_denmark.RDS"
    candidate_paths = [
        paths.raw_dir / filename,
        paths.data_dir / filename,
        Path(filename),
    ]
    file_path = next((p for p in candidate_paths if p.exists()), None)
    if file_path is None:
        raise FileNotFoundError(f"Could not find corpus file {filename} in {candidate_paths}")

    raw_df = load_raw_dataframe(file_path)
    normalized = normalize_corpus(raw_df, corpus_cfg)
    validated, issues = validate_corpus_df(normalized, schema)
    return validated, issues
