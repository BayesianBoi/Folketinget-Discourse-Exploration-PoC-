from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class TopicTerm(BaseModel):
    term: str
    weight: float


class Topic(BaseModel):
    topic_id: int
    label: str
    terms: List[TopicTerm]


class PrevalencePoint(BaseModel):
    party: str
    time_bin: str
    topic_id: int
    mean_weight: float
    doc_count: int


class SentimentPoint(BaseModel):
    party: str
    time_bin: str
    area: str
    mean_sentiment: float
    doc_count: int


class TopicsResponse(BaseModel):
    topics: List[Topic]
    metadata: Optional[dict] = None


class PrevalenceResponse(BaseModel):
    points: List[PrevalencePoint]


class SentimentResponse(BaseModel):
    points: List[SentimentPoint]
