import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.analysis import (
    aggregate_sentiment_by_area,
    filter_doc_topics_by_area,
    match_topics_to_areas,
    melt_doc_topics,
    resolve_provider,
)
from dk_politics_topics.utils.logging import get_logger, setup_logging


def main() -> None:
    setup_logging()
    cfg = DEFAULT_CONFIG
    cfg.ensure()
    logger = get_logger(__name__)

    topics_path = cfg.paths.exports_dir / cfg.export.topic_json
    doc_topics_path = cfg.paths.exports_dir / cfg.export.doc_topics_parquet
    data_path = cfg.paths.processed_dir / "preprocessed.parquet"
    if not all([topics_path.exists(), doc_topics_path.exists(), data_path.exists()]):
        raise FileNotFoundError("Ensure steps 01-03 have been run before sentiment analysis.")

    topics_payload = json.loads(topics_path.read_text(encoding="utf-8"))
    topic_terms_raw = topics_payload.get("topics", {})
    labels = topics_payload.get("metadata", {}).get("labels", {})
    topic_terms = {int(k): v for k, v in topic_terms_raw.items()}
    matches = match_topics_to_areas(topic_terms, labels=labels)

    doc_topics = pd.read_parquet(doc_topics_path)
    long_doc_topics = melt_doc_topics(doc_topics)
    metadata = pd.read_parquet(data_path)[["doc_id", "party", "time_bin", "text"]]

    provider = resolve_provider(cfg.sentiment)

    # Score all relevant documents once and save
    logger.info("Scoring documents for sentiment...")
    # Get all unique doc_ids appearing in any area
    relevant_docs = long_doc_topics[long_doc_topics["topic_id"].isin([t for v in matches.values() for t in v])]["doc_id"].unique()
    
    # Or simpler: just score the metadata subset if feasible, or all metadata if we want global sentiment
    # Let's score all metadata to support the global/bespoke plots later?
    # Metadata has all docs from preprocessed.
    df_scored = score_documents(metadata, "text", provider)
    
    # Save scores
    sentiment_path = cfg.paths.exports_dir / "sentiment_scores.parquet"
    df_scored[["doc_id", "sentiment"]].to_parquet(sentiment_path, index=False)
    logger.info("Saved document sentiment scores to %s", sentiment_path)

    frames = []
    for area, topic_ids in matches.items():
        if not topic_ids:
            logger.warning("No topics matched for area %s", area)
            continue
        filtered = filter_doc_topics_by_area(long_doc_topics, topic_ids, area)
        
        # Merge scores back for aggregation
        merged = filtered.merge(df_scored[["doc_id", "party", "time_bin", "sentiment"]], on="doc_id", how="left")
        
        agg = (
            merged.groupby(["party", "time_bin"])
            .agg(mean_sentiment=("sentiment", "mean"), doc_count=("doc_id", "count"))
            .reset_index()
        )
        agg["area"] = area # Add area column
        frames.append(agg)

    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(columns=["party", "time_bin", "area", "mean_sentiment", "doc_count"])

    output_path = cfg.paths.exports_dir / cfg.export.controversial_parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info("Saved controversial sentiment to %s", output_path)


if __name__ == "__main__":
    main()
