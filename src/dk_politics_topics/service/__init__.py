"""Service layer for API-ready functions (no web framework yet)."""

from .repository import ArtifactRepository
from .endpoints import (
    get_parties,
    get_topics,
    get_prevalence,
    compare_parties,
    get_controversial_sentiment,
)

__all__ = [
    "ArtifactRepository",
    "get_parties",
    "get_topics",
    "get_prevalence",
    "compare_parties",
    "get_controversial_sentiment",
]
