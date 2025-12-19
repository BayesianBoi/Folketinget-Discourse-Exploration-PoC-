from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import pandas as pd

from ..config import SentimentConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SentimentProvider:
    def score(self, text: str) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class TransformersSentiment(SentimentProvider):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str = "cpu",
        neutral_threshold: float = 0.0,
    ):
        from transformers import pipeline
        import torch
        
        # Resolve device
        if device == "auto":
            if torch.backends.mps.is_available():
                device_id = "mps"
            elif torch.cuda.is_available():
                device_id = 0
            else:
                device_id = -1
        else:
            device_id = 0 if device == "cuda" else -1
            if device == "mps":
                 device_id = "mps"

        self.batch_size = batch_size
        self.neutral_threshold = float(neutral_threshold or 0.0)
        logger.info(f"Loading sentiment model '{model_name}' on device={device_id} (batch_size={batch_size})")
        
        # Initialize pipeline with truncation to handle long speeches.
        # Prefer returning *all* class scores so we can compute a calibrated polarity:
        #   sentiment = P(positive) - P(negative)
        # This tends to be much less biased than using only the argmax label confidence.
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device_id,
                truncation=True,
                max_length=512,
                top_k=None,
            )
        except TypeError:  # older transformers
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device_id,
                truncation=True,
                max_length=512,
                return_all_scores=True,
            )

    def score(self, text: str) -> float:
        # Fallback for single-text usage (deprecated in favor of batch scoring)
        return self.score_batch([text])[0]

    @staticmethod
    def _label_bucket(label: str) -> str:
        l = (label or "").lower()
        if "positiv" in l or "positive" in l:
            return "pos"
        if "negativ" in l or "negative" in l:
            return "neg"
        if "neutral" in l:
            return "neu"
        return "other"

    def _score_from_distribution(self, dist: Sequence[Dict[str, float]]) -> float:
        pos = 0.0
        neg = 0.0
        for item in dist:
            bucket = self._label_bucket(str(item.get("label", "")))
            score = float(item.get("score", 0.0))
            if bucket == "pos":
                pos += score
            elif bucket == "neg":
                neg += score
        # If we can't recognize labels, fall back to neutral.
        if pos == 0.0 and neg == 0.0:
            return 0.0
        # Range is naturally [-1, 1] if the dist sums to 1.
        score = max(-1.0, min(1.0, pos - neg))
        if abs(score) < self.neutral_threshold:
            return 0.0
        return score

    def _score_from_top_label(self, out: Dict[str, float]) -> float:
        bucket = self._label_bucket(str(out.get("label", "")))
        score = float(out.get("score", 0.0))
        if bucket == "pos":
            signed = score
        elif bucket == "neg":
            signed = -score
        else:
            signed = 0.0
        if abs(signed) < self.neutral_threshold:
            return 0.0
        return signed

    def score_batch(self, texts: Iterable[str]) -> List[float]:
        # Handle empty/None
        clean_texts = [t if isinstance(t, str) and t.strip() else "" for t in texts]
        
        # Process in batches via pipeline
        results = []
        # The pipeline is iterable for efficient batching
        for out in self.pipeline(clean_texts, batch_size=self.batch_size):
            # When configured with top_k=None / return_all_scores=True, each `out`
            # is a list of {"label","score"} dicts (one per class).
            if isinstance(out, list):
                results.append(self._score_from_distribution(out))
            else:
                results.append(self._score_from_top_label(out))
        return results


def resolve_provider(config: SentimentConfig) -> SentimentProvider:
    if config.approach != "huggingface":
        raise ValueError(
            "Only transformer-based sentiment is supported. "
            "Set `cfg.sentiment.approach = \"huggingface\"`."
        )
    if not config.huggingface_model:
        raise ValueError("SentimentConfig.huggingface_model must be set.")

    try:
        return TransformersSentiment(
            model_name=config.huggingface_model,
            batch_size=config.batch_size,
            device=config.device,
            neutral_threshold=config.neutral_threshold,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize HuggingFace sentiment pipeline for '{config.huggingface_model}'. "
            "Ensure the model is available in the local HF cache or run with network access enabled."
        ) from exc


def score_documents(
    df: pd.DataFrame,
    text_col: str,
    provider: SentimentProvider,
) -> pd.DataFrame:
    scored = df.copy()
    texts = scored[text_col].fillna("").astype(str).tolist()
    
    if hasattr(provider, "score_batch"):
        logger.info(f"Scoring {len(texts)} documents in batches...")
        scores = provider.score_batch(texts)
    else:
        scores = [provider.score(t) for t in texts]
        
    scored["sentiment"] = scores
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
