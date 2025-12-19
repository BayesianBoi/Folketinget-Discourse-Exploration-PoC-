from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from ..config import PathsConfig, TopeaxConfig
from ..utils.cache import save_json
from ..utils.logging import get_logger
from ..utils.randomness import set_seed

logger = get_logger(__name__)

try:  # turftopic is optional for testing; fallback LDA will be used otherwise
    from turftopic import Topeax  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Topeax = None


class Heartbeat:
    """Background logger that emits a heartbeat while long operations run."""

    def __init__(self, message: str, interval: int = 60):
        self.message = message
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:  # pragma: no cover - timing utility
        while not self._stop.wait(self.interval):
            logger.info(self.message)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join(timeout=1)


class TopeaxModelRunner:
    def __init__(self, cfg: TopeaxConfig, paths: PathsConfig):
        self.cfg = cfg
        self.paths = paths
        self.model = None
        self.topic_terms: Dict[int, List[Dict[str, float]]] = {}

    def _fit_fallback(self, texts: List[str]) -> Tuple[pd.DataFrame, Dict[int, List[Dict[str, float]]]]:
        vectorizer = CountVectorizer(
            stop_words=None,
            min_df=self.cfg.min_df,
            max_features=self.cfg.max_features,
        )
        doc_term = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(
            n_components=min(10, doc_term.shape[0]),
            random_state=self.cfg.random_state,
            learning_method="online",
        )
        doc_topic = lda.fit_transform(doc_term)
        feature_names = vectorizer.get_feature_names_out()
        topic_terms: Dict[int, List[Dict[str, float]]] = {}
        for idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-self.cfg.top_k_terms :][::-1]
            topic_terms[idx] = [
                {"term": feature_names[i], "weight": float(topic[i])} for i in top_indices
            ]
        self.model = lda
        return pd.DataFrame(doc_topic), topic_terms

    def _extract_terms_from_model(self, model: object) -> Dict[int, List[Dict[str, float]]]:
        topic_terms: Dict[int, List[Dict[str, float]]] = {}

        def _coerce_weight(val) -> float:
            try:
                return float(val)
            except Exception:
                if isinstance(val, (list, tuple)) and val:
                    try:
                        return float(val[0])
                    except Exception:
                        return 0.0
                return 0.0

        if hasattr(model, "get_topics"):
            try:
                raw_topics = model.get_topics(top_k=self.cfg.top_k_terms)
            except TypeError:
                raw_topics = model.get_topics()
            if isinstance(raw_topics, dict):
                for idx, words in raw_topics.items():
                    topic_terms[int(idx)] = [
                        {"term": w[0], "weight": _coerce_weight(w[1])} if isinstance(w, (list, tuple)) else {"term": str(w), "weight": 0.0}
                        for w in words[: self.cfg.top_k_terms]
                    ]
            elif isinstance(raw_topics, list):
                for item in raw_topics:
                    # Turftopic (at least some versions/models) returns [(topic_id, [(term, weight), ...]), ...]
                    if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], list):
                        topic_idx = int(item[0])
                        words = item[1]
                        
                        topic_terms[topic_idx] = [
                            {"term": w[0], "weight": _coerce_weight(w[1])} 
                            if isinstance(w, (list, tuple)) else {"term": str(w), "weight": 0.0}
                            for w in words[: self.cfg.top_k_terms]
                        ]
                    else:
                        # Fallback for simpler lists of lists: [[(term, weight), ...], ...]
                        # Here 'item' IS 'words'
                        # We guess the index based on enumeration or just implicit order?
                        # The original loop used enumerate. Let's restore that for generic lists of lists.
                        pass
                
                # If we didn't populate topic_terms via the structure above (e.g. it was just [[terms], [terms]]),
                # retry with simple enumeration.
                if not topic_terms:
                    for idx, words in enumerate(raw_topics):
                         topic_terms[idx] = [
                            {"term": w[0], "weight": _coerce_weight(w[1])} 
                            if isinstance(w, (list, tuple)) else {"term": str(w), "weight": 0.0}
                            for w in words[: self.cfg.top_k_terms]
                        ]
        elif hasattr(model, "topic_words_"):
            for idx, words in enumerate(getattr(model, "topic_words_")):
                topic_terms[idx] = [{"term": w, "weight": 0.0} for w in words[: self.cfg.top_k_terms]]
        return topic_terms

    def fit(
        self,
        texts: Iterable[str],
        doc_ids: Iterable[str],
        embeddings: np.ndarray | None = None,
    ) -> Tuple[pd.DataFrame, Dict[int, List[Dict[str, float]]]]:
        set_seed(self.cfg.random_state)
        texts_list = list(texts)
        doc_ids_list = list(doc_ids)

        if Topeax is None:
            raise ImportError(
                "turftopic is not installed or failed to import. "
                "The pipeline requires turftopic.Topeax to proceed. "
                "Please install it via 'pip install turftopic' or check your environment."
            )
        
        logger.info("Fitting Topeax with perplexity=%s", self.cfg.perplexity)
        
        # Use CountVectorizer with our stopwords and config
        vectorizer = CountVectorizer(
            stop_words=self.cfg.stopwords,
            min_df=self.cfg.min_df,
            max_features=self.cfg.max_features,
        )
        
        model_kwargs = {
            "perplexity": self.cfg.perplexity,
            "random_state": self.cfg.random_state,
            "vectorizer": vectorizer,
        }
        if self.cfg.embedding_model:
            model_kwargs["encoder"] = self.cfg.embedding_model
        
        model = Topeax(**model_kwargs)
        with Heartbeat("... Topeax still fitting; this step can take several minutes", interval=60):
            try:
                result = model.fit_transform(texts_list, embeddings=embeddings)
            except TypeError:
                result = model.fit_transform(texts_list)
        
        # LLM Topic Naming
        if self.cfg.openai_model:
            try:
                from turftopic.analyzers import OpenAIAnalyzer
                logger.info("Renaming topics using OpenAI model: %s", self.cfg.openai_model)
                analyzer = OpenAIAnalyzer(model_name=self.cfg.openai_model)
                model.rename_topics(analyzer)
            except Exception as e:
                logger.warning(f"Failed to name topics with OpenAI: {e}")
        
        # turftopic may return (topics, matrix[, extra]) or just a matrix
        doc_topic_matrix = None
        if hasattr(model, "doc_topic_") and model.doc_topic_ is not None:
            doc_topic_matrix = model.doc_topic_
        elif isinstance(result, tuple) and len(result) >= 2:
            doc_topic_matrix = result[1]
        elif isinstance(result, np.ndarray):
            doc_topic_matrix = result

        if doc_topic_matrix is None:
            raise ValueError("Topeax did not produce a document-topic matrix.")

        doc_topic_df = pd.DataFrame(doc_topic_matrix)
        topic_terms = self._extract_terms_from_model(model)
        self.model = model

        if not topic_terms:
            topic_terms = self._extract_terms_from_model(self.model)

        if len(doc_topic_df) != len(doc_ids_list):
            raise ValueError("doc_ids length does not match doc-topic matrix rows")
        doc_topic_df.insert(0, "doc_id", doc_ids_list)
        self.topic_terms = topic_terms
        return doc_topic_df, topic_terms

    def save_model(self, name: str = "topeax_model.joblib") -> Path:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.models_dir / name
        import joblib  # local import to keep dependency optional

        joblib.dump(self.model, path)
        logger.info("Saved model to %s", path)
        return path

    def save_topics(self, path: Path, metadata: Dict | None = None) -> None:
        payload = {
            "topics": self.topic_terms,
            "metadata": metadata or {},
        }
        save_json(payload, path)
