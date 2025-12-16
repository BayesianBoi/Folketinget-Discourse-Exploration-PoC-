from __future__ import annotations

from typing import Dict, Iterable, List, Set

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SEEDS: Dict[str, List[str]] = {
    "immigration": [
        "immigration",
        "asyl",
        "integration",
        "integrations",
        "flygtning",
        "flygtninge",
        "indvandring",
        "udlænding",
        "udlændingepolitik",
        "grænsekontrol",
        "ghetto",
    ],
    "taxation": [
        "skat",
        "beskatning",
        "afgift",
        "afgifter",
        "skattereform",
        "moms",
        "topskat",
        "selskabsskat",
        "personskat",
        "skatteprovenu",
    ],
    "climate": [
        "klima",
        "klimaforandring",
        "co2",
        "miljø",
        "energi",
        "grøn",
        "bæredygtig",
        "klimakrise",
        "vindmølle",
        "solenergi",
        "udledning",
        "klimamål",
    ],
}


def match_topics_to_areas(
    topic_terms: Dict[int, List[Dict[str, float]]],
    seeds: Dict[str, List[str]] | None = None,
    labels: Dict[str, str] | None = None,
) -> Dict[str, List[int]]:
    seeds = seeds or DEFAULT_SEEDS
    matches: Dict[str, List[int]] = {k: [] for k in seeds}
    for topic_id, terms in topic_terms.items():
        cleaned_terms = []
        for t in terms:
            term_val = t.get("term") if isinstance(t, dict) else t
            if isinstance(term_val, (list, tuple)) and term_val:
                term_val = term_val[0]
            if isinstance(term_val, str):
                cleaned_terms.append(term_val.lower())
        label_text = labels.get(str(topic_id), "") if labels else ""
        lower_terms = " ".join(cleaned_terms + [label_text.lower()])
        for area, keywords in seeds.items():
            if any(keyword.lower() in lower_terms for keyword in keywords):
                matches[area].append(topic_id)
    logger.info("Matched topics to controversial areas: %s", {k: len(v) for k, v in matches.items()})
    return matches


def filter_doc_topics_by_area(
    long_doc_topics: pd.DataFrame, topic_ids: Iterable[int], area: str
) -> pd.DataFrame:
    topic_set: Set[int] = set(topic_ids)
    filtered = long_doc_topics[long_doc_topics["topic_id"].isin(topic_set)].copy()
    filtered["area"] = area
    return filtered
