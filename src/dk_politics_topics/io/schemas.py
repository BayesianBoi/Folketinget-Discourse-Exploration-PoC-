from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CorpusSchema:
    required_columns: Tuple[str, ...] = ("doc_id", "date", "party", "text")
    optional_columns: Tuple[str, ...] = ("year", "source")

    def all_columns(self) -> Tuple[str, ...]:
        return self.required_columns + self.optional_columns


def validate_corpus_df(df: pd.DataFrame, schema: CorpusSchema | None = None) -> Tuple[pd.DataFrame, List[str]]:
    schema = schema or CorpusSchema()
    issues: List[str] = []

    initial_duplicates = 0
    if "doc_id" in df.columns:
        initial_duplicates = int(df["doc_id"].duplicated().sum())

    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        issues.append(f"Mangler kolonner: {missing}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        missing_dates = df["date"].isna().sum()
        if missing_dates:
            issues.append(f"{missing_dates} rækker mangler gyldig dato")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    elif "date" in df.columns:
        df["year"] = df["date"].dt.year

    if "text" in df.columns:
        too_short = df["text"].fillna("").str.len() < 5
        if too_short.any():
            issues.append(f"{too_short.sum()} tekster er for korte (<5 tegn)")
            df = df.loc[~too_short].copy()

    if "doc_id" in df.columns:
        if initial_duplicates:
            issues.append(f"{initial_duplicates} duplikerede doc_id fjernet")
        df = df.drop_duplicates(subset="doc_id").copy()

    unexpected_cols = [c for c in df.columns if c not in schema.all_columns()]
    if unexpected_cols:
        issues.append(f"Uventede kolonner bevaret: {unexpected_cols}")

    logger.info("Validering fuldført med %d issues", len(issues))
    return df, issues
