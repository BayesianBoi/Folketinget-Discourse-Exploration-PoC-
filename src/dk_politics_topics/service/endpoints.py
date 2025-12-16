from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd

from .api_models import (
    PrevalencePoint,
    PrevalenceResponse,
    SentimentPoint,
    SentimentResponse,
    Topic,
    TopicTerm,
    TopicsResponse,
)
from .repository import ArtifactRepository
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _filter_by_timerange(df: pd.DataFrame, time_col: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df[df[time_col] >= start]
    if end:
        df = df[df[time_col] <= end]
    return df


def get_parties(repo: ArtifactRepository) -> List[str]:
    return repo.list_parties()


def get_topics(repo: ArtifactRepository) -> TopicsResponse:
    data = repo.load_topics()
    topics_raw = data.get("topics", {})
    labels = data.get("metadata", {}).get("labels", {})
    topics: List[Topic] = []
    for topic_id, terms in topics_raw.items():
        topic_id_int = int(topic_id)
        term_objs = [TopicTerm(term=t.get("term"), weight=t.get("weight", 0.0)) for t in terms]
        topics.append(Topic(topic_id=topic_id_int, label=labels.get(str(topic_id_int), ""), terms=term_objs))
    return TopicsResponse(topics=topics, metadata=data.get("metadata"))


def get_prevalence(
    repo: ArtifactRepository,
    party: Optional[str] = None,
    topic_id: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    time_bin_col: str = "year",
) -> PrevalenceResponse:
    df = repo.load_prevalence()
    if party:
        df = df[df["party"] == party]
    if topic_id is not None:
        df = df[df["topic_id"] == topic_id]
    df = _filter_by_timerange(df, time_bin_col, start, end)
    points = [
        PrevalencePoint(
            party=row["party"],
            time_bin=str(row[time_bin_col]),
            topic_id=int(row["topic_id"]),
            mean_weight=float(row["mean_weight"]),
            doc_count=int(row["doc_count"]),
        )
        for _, row in df.iterrows()
    ]
    return PrevalenceResponse(points=points)


def compare_parties(
    repo: ArtifactRepository,
    party_a: str,
    party_b: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    time_bin_col: str = "year",
) -> PrevalenceResponse:
    df = repo.load_prevalence()
    df = df[df["party"].isin([party_a, party_b])]
    df = _filter_by_timerange(df, time_bin_col, start, end)
    points = [
        PrevalencePoint(
            party=row["party"],
            time_bin=str(row[time_bin_col]),
            topic_id=int(row["topic_id"]),
            mean_weight=float(row["mean_weight"]),
            doc_count=int(row["doc_count"]),
        )
        for _, row in df.iterrows()
    ]
    return PrevalenceResponse(points=points)


def get_controversial_sentiment(
    repo: ArtifactRepository,
    area: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    time_bin_col: str = "year",
) -> SentimentResponse:
    df = repo.load_controversial_sentiment()
    if area:
        df = df[df["area"] == area]
    df = _filter_by_timerange(df, time_bin_col, start, end)
    points = [
        SentimentPoint(
            party=row["party"],
            time_bin=str(row[time_bin_col]),
            area=row["area"],
            mean_sentiment=float(row["mean_sentiment"]),
            doc_count=int(row["doc_count"]),
        )
        for _, row in df.iterrows()
    ]
    return SentimentResponse(points=points)
