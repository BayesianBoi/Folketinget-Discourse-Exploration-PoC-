import re
from typing import Iterable

import pandas as pd

from ..config import PreprocessConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


def clean_text(text: str, cfg: PreprocessConfig) -> str:
    """Lightweight cleaning that keeps Danish characters intact."""
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("\xa0", " ")
    for phrase in cfg.remove_boilerplate_phrases:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cfg.lowercase:
        cleaned = cleaned.lower()
    return cleaned


def clean_dataframe(df: pd.DataFrame, text_col: str, cfg: PreprocessConfig) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Filter by Speaker (if column exists)
    if "speaker" in df.columns and cfg.ignored_speakers:
        original_count = len(df)
        df = df[~df["speaker"].isin(cfg.ignored_speakers)]
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info("Removed %d speeches by ignored speakers (e.g., Formanden)", filtered_count)

    # Filter by minimum length (words)
    if hasattr(cfg, 'min_words') and cfg.min_words > 0:
        original_count = len(df)
        # Use the text_col for word count calculation
        df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))
        df = df[df["word_count"] >= cfg.min_words].drop(columns=["word_count"])
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.warning("Removed %d texts with fewer than %d words", filtered_count, cfg.min_words)

    # Filter by ignored parties
    if "party" in df.columns and hasattr(cfg, 'ignored_parties') and cfg.ignored_parties:
        original_count = len(df)
        df = df[~df["party"].isin(cfg.ignored_parties)]
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info("Removed %d speeches by ignored parties", filtered_count)

    # 2. Clean Text
    df[text_col] = df[text_col].fillna("").astype(str).apply(lambda t: clean_text(t, cfg))
    
    # 3. Filter by Length (Word Count & Char Length)
    # Peer study used "one word" removal. We use min_words (default 5) for robustness.
    # Simple split by whitespace is sufficient for this check.
    word_counts = df[text_col].str.split().str.len()
    too_short_words = word_counts < cfg.min_words
    
    # Also keep the char length check if it was useful, or rely on words. 
    # Let's trust words more for "content".
    
    if too_short_words.any():
        logger.warning("Removing %d texts with fewer than %d words", too_short_words.sum(), cfg.min_words)
        df = df.loc[~too_short_words].copy()
        
    return df


def filter_parties_by_min_share(df: pd.DataFrame, party_col: str, min_share: float) -> pd.DataFrame:
    """Drop parties that make up less than `min_share` of rows.

    Intended as a corpus-level filter to reduce noise from very small parties.
    """
    if min_share is None or float(min_share) <= 0:
        return df
    if party_col not in df.columns:
        return df

    shares = df[party_col].value_counts(normalize=True, dropna=False)
    keep = shares[shares >= float(min_share)].index
    removed = shares[shares < float(min_share)]
    if removed.empty:
        return df

    before = len(df)
    df = df[df[party_col].isin(keep)].copy()
    after = len(df)
    removed_parties = ", ".join(f"{p} ({s:.2%})" for p, s in removed.items())
    logger.info(
        "Removed %d speeches from parties below %.2f%% share: %s",
        before - after,
        float(min_share) * 100.0,
        removed_parties,
    )
    return df
