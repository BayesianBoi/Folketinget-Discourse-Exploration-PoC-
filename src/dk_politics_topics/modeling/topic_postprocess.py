from __future__ import annotations

from typing import Dict, List

import pandas as pd


def label_topics(topic_terms: Dict[int, List[Dict[str, float]]], top_n: int = 3) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for topic_id, terms in topic_terms.items():
        words = []
        for t in terms[:top_n]:
            term_val = t.get("term") if isinstance(t, dict) else t
            # Handle cases where term_val may be a tuple/list from upstream
            if isinstance(term_val, (list, tuple)) and term_val:
                term_val = term_val[0]
            if isinstance(term_val, str):
                words.append(term_val)
        labels[topic_id] = ", ".join(words)
    return labels


def topic_terms_to_df(topic_terms: Dict[int, List[Dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for topic_id, terms in topic_terms.items():
        for rank, term in enumerate(terms):
            rows.append(
                {
                    "topic_id": topic_id,
                    "term": term.get("term"),
                    "weight": term.get("weight", 0.0),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)
