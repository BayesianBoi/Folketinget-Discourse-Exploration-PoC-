import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.utils.cache import save_json
from dk_politics_topics.utils.logging import get_logger, setup_logging
from dk_politics_topics.viz import (
    plot_controversial_sentiment,
    plot_party_comparison,
    plot_prevalence_over_time,
)


def main() -> None:
    setup_logging()
    cfg = DEFAULT_CONFIG
    cfg.ensure()
    logger = get_logger(__name__)

    processed_path = cfg.paths.processed_dir / "preprocessed.parquet"
    prevalence_path = cfg.paths.exports_dir / cfg.export.prevalence_parquet
    controversial_path = cfg.paths.exports_dir / cfg.export.controversial_parquet
    doc_topics_path = cfg.paths.exports_dir / cfg.export.doc_topics_parquet
    topics_json_path = cfg.paths.exports_dir / cfg.export.topic_json

    df = pd.read_parquet(processed_path)
    prevalence = pd.read_parquet(prevalence_path)
    controversial = pd.read_parquet(controversial_path)
    doc_topics = pd.read_parquet(doc_topics_path)
    import json
    topics_payload = json.loads(topics_json_path.read_text(encoding="utf-8"))

    metadata = {
        "config": cfg.as_dict(),
        "corpus": {
            "rows": len(df),
            "parties": sorted(df["party"].dropna().unique().tolist()),
            "date_min": df["date"].min().isoformat() if "date" in df else None,
            "date_max": df["date"].max().isoformat() if "date" in df else None,
        },
    }
    metadata_path = cfg.paths.exports_dir / cfg.export.metadata_json
    save_json(metadata, metadata_path)

    labels = topics_payload.get("metadata", {}).get("labels", {})
    topic_terms = topics_payload.get("topics", {})

    # Additional interpretable exports (legacy-style)
    topic_cols = [c for c in doc_topics.columns if c != "doc_id"]
    top_topic = doc_topics[topic_cols].idxmax(axis=1)
    top_topic = top_topic.apply(lambda x: int(str(x).replace("topic_", "")) if isinstance(x, str) else int(x))
    topic_counts = top_topic.value_counts().to_dict()

    topic_details_rows = []
    for topic_id_str, terms in topic_terms.items():
        topic_id = int(topic_id_str)
        label = labels.get(str(topic_id), "")
        # Robust term extraction handling malformed lists in 'term' field
        extracted_terms = []
        for t in terms[:10]:
            if isinstance(t, dict):
                raw_term = t.get("term")
                if isinstance(raw_term, list) and raw_term:
                    extracted_terms.append(str(raw_term[0]))
                else:
                    extracted_terms.append(str(raw_term))
            else:
                extracted_terms.append(str(t))
        top_terms = ", ".join(extracted_terms)
        topic_details_rows.append(
            {
                "topic_id": topic_id,
                "label": label,
                "doc_count": topic_counts.get(topic_id, 0),
                "top_terms": top_terms,
            }
        )
    topic_details_df = pd.DataFrame(topic_details_rows).sort_values(by="doc_count", ascending=False)
    topic_details_df.to_csv(cfg.paths.exports_dir / "topic_details.csv", index=False)

    df_top = pd.DataFrame({"doc_id": doc_topics["doc_id"], "top_topic": top_topic})
    df_top = df_top.merge(df[["doc_id", "party"]], on="doc_id", how="left")
    party_topic_matrix = df_top.pivot_table(index="party", columns="top_topic", aggfunc="size", fill_value=0)
    party_topic_matrix.columns = [
        f"{col} | {labels.get(str(int(col)), '')}" for col in party_topic_matrix.columns
    ]
    party_topic_matrix.to_csv(cfg.paths.exports_dir / "party_topic_matrix.csv")

    summary_path = cfg.paths.exports_dir / "summary_statistics.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("DANISH PARLIAMENT TOPIC ANALYSIS (Topeax)\n")
        f.write("=" * 70 + "\n\n")
        f.write("MODEL\n")
        f.write(f"  Topics: {len(topic_terms)}\n")
        f.write(f"  Encoder: {cfg.embedding.model_name}\n")
        f.write(f"  Perplexity: {cfg.topeax.perplexity}\n\n")
        f.write("TOP TOPICS (by doc_count)\n")
        for _, row in topic_details_df.head(10).iterrows():
            f.write(f"  {int(row['topic_id']):2d} {row['label']}: {int(row['doc_count']):,}\n")
        f.write("\nPARTY COVERAGE (docs per party)\n")
        party_counts = df["party"].value_counts()
        for party, count in party_counts.items():
            f.write(f"  {party}: {count:,}\n")

    if len(prevalence):
        plot_prevalence_over_time(
            prevalence,
            output_path=cfg.paths.plots_dir / "prevalence_over_time.html",
            party=None,
            time_bin_col="time_bin",
        )
        plot_party_comparison(
            prevalence,
            output_path=cfg.paths.plots_dir / "party_comparison.html",
            time_bin_col="time_bin",
        )
    else:
        logger.warning("Prevalence dataframe is empty; skipping prevalence plots.")

    if len(controversial):
        plot_controversial_sentiment(
            controversial,
            output_path=cfg.paths.plots_dir / "controversial_sentiment.html",
            time_bin_col="time_bin",
        )
    else:
        logger.warning("Controversial sentiment dataframe is empty; skipping sentiment plots.")

    logger.info("Export build complete")


if __name__ == "__main__":
    main()
