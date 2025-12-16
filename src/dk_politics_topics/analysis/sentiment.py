from __future__ import annotations

import re
from typing import Dict, Iterable

import pandas as pd

from ..config import SentimentConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SentimentProvider:
    def score(self, text: str) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class LexiconSentiment(SentimentProvider):
    def __init__(self, config: SentimentConfig):
        self.positive = set(word.lower() for word in config.lexicon_positive)
        self.negative = set(word.lower() for word in config.lexicon_negative)
        self.neutral_threshold = config.neutral_threshold

    def score(self, text: str) -> float:
        tokens = re.findall(r"\b\w+\b", text.lower())
        pos = sum(token in self.positive for token in tokens)
        neg = sum(token in self.negative for token in tokens)
        if pos + neg == 0:
            return 0.0
        score = (pos - neg) / (pos + neg)
        if abs(score) < self.neutral_threshold:
            return 0.0
        return score


class TransformersSentiment(SentimentProvider):  # pragma: no cover - optional
    def __init__(self, model_name: str):
        from transformers import pipeline

        self.pipeline = pipeline("sentiment-analysis", model=model_name)

    def score(self, text: str) -> float:
        result = self.pipeline(text)[0]
        label = result["label"].lower()
        score = result.get("score", 0.0)
        return score if "pos" in label else -score


def resolve_provider(config: SentimentConfig) -> SentimentProvider:
    if config.approach == "huggingface" and config.huggingface_model:
        try:
            return TransformersSentiment(config.huggingface_model)
        except Exception as exc:
            logger.warning("Falling back to lexicon sentiment: %s", exc)
    return LexiconSentiment(config)


def score_documents(
    df: pd.DataFrame,
    text_col: str,
    provider: SentimentProvider,
) -> pd.DataFrame:
    scored = df.copy()
    scored["sentiment"] = scored[text_col].fillna("").astype(str).apply(provider.score)
    return scored


def aggregate_sentiment_by_area(
    filtered_docs: pd.DataFrame,
    metadata: pd.DataFrame,
    provider: SentimentProvider,
    time_bin_col: str = "year",
) -> pd.DataFrame:
    merged = filtered_docs.merge(metadata[["doc_id", "party", time_bin_col, "text"]], on="doc_id", how="left")
    merged = score_documents(merged, "text", provider)
    grouped = (
        merged.groupby(["party", time_bin_col, "area"])
        .agg(mean_sentiment=("sentiment", "mean"), doc_count=("doc_id", "count"))
        .reset_index()
    )
    return grouped
