from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def melt_doc_topics(doc_topics: pd.DataFrame) -> pd.DataFrame:
    topic_cols = [c for c in doc_topics.columns if c != "doc_id"]
    long_df = doc_topics.melt(id_vars="doc_id", value_vars=topic_cols, var_name="topic_id", value_name="weight")
    long_df["topic_id"] = long_df["topic_id"].apply(_coerce_topic_id)
    return long_df


def _coerce_topic_id(value) -> int:
    string_value = str(value)
    if string_value.startswith("topic_"):
        string_value = string_value.replace("topic_", "")
    try:
        return int(string_value)
    except ValueError:
        return -1


def compute_prevalence(
    doc_topics: pd.DataFrame,
    metadata: pd.DataFrame,
    time_bin_col: str = "year",
) -> pd.DataFrame:
    long_df = melt_doc_topics(doc_topics)
    merged = long_df.merge(metadata[["doc_id", "party", time_bin_col]], on="doc_id", how="left")

    grouped = (
        merged.groupby(["party", time_bin_col, "topic_id"])
        .agg(mean_weight=("weight", "mean"), doc_count=("doc_id", "count"))
        .reset_index()
    )
    logger.info("Computed prevalence with %d rows", len(grouped))
    return grouped
