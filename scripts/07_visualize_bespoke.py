import sys
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.utils.logging import setup_logging, get_logger
from dk_politics_topics.analysis.sentiment import resolve_provider, score_documents

def main():
    setup_logging()
    logger = get_logger(__name__)
    cfg = DEFAULT_CONFIG
    
    # 1. Load Data
    logger.info("Loading data...")
    processed_path = cfg.paths.processed_dir / "preprocessed.parquet"
    doc_topics_path = cfg.paths.exports_dir / cfg.export.doc_topics_parquet
    topics_json_path = cfg.paths.exports_dir / cfg.export.topic_json
    
    if not all(p.exists() for p in [processed_path, doc_topics_path, topics_json_path]):
        logger.error("Missing required input files. Run the pipeline first.")
        return

    df_docs = pd.read_parquet(processed_path)
    df_topics = pd.read_parquet(doc_topics_path)
    topics_payload = json.loads(topics_json_path.read_text(encoding="utf-8"))
    
    # Get labels
    labels_map = topics_payload.get("metadata", {}).get("labels", {})
    def get_label(tid):
        return f"{tid}: {labels_map.get(str(tid), f'Topic {tid}')}"

    # 2. Merge Data
    logger.info("Merging and processing data...")
    
    # Extract top topic for each document
    topic_cols = [c for c in df_topics.columns if c != "doc_id"]
    # Assuming standard format, we find max col
    df_topics["top_topic_id"] = df_topics[topic_cols].idxmax(axis=1).apply(
        lambda x: int(x.split("_")[-1]) if isinstance(x, str) else int(x)
    )
    
    # Merge back to metadata
    df_merged = df_docs.merge(df_topics[["doc_id", "top_topic_id"]], on="doc_id", how="inner")
    df_merged["topic_label"] = df_merged["top_topic_id"].apply(get_label)
    
    # Ensure year column exists
    if "year" not in df_merged.columns:
         if "date" in df_merged.columns:
             try:
                df_merged["year"] = pd.to_datetime(df_merged["date"]).dt.year
             except:
                logger.warning("Could not parse date for year. Using 'time_bin' if available.")
                if "time_bin" in df_merged.columns:
                     df_merged["year"] = df_merged["time_bin"] # Fallback
    
    # Filter for reasonable years (optional, but good for clean plots)
    df_merged = df_merged[df_merged["year"].notna()]
    
    output_dir = cfg.paths.plots_dir / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # PLOT 1: Overall Topic Prevalence Over Time
    # ---------------------------------------------------------
    logger.info("Generating Plot 1: Overall Topic Prevalence...")
    prevalence_overall = df_merged.groupby(["year", "topic_label"]).size().reset_index(name="count")
    # Normalize by year total
    year_totals = df_merged.groupby("year").size().reset_index(name="year_total")
    prevalence_overall = prevalence_overall.merge(year_totals, on="year")
    prevalence_overall["prevalence"] = prevalence_overall["count"] / prevalence_overall["year_total"]
    
    fig1 = px.line(
        prevalence_overall, 
        x="year", 
        y="prevalence", 
        color="topic_label",
        title="Topic Prevalence Over Time (All Parties)",
        labels={"prevalence": "Share of Speeches", "year": "Year", "topic_label": "Topic"},
        markers=True
    )
    fig1.update_layout(hovermode="x unified")
    fig1.write_html(output_dir / "01_topic_prevalence_overall.html")
    
    # ---------------------------------------------------------
    # PLOT 2: Topic Prevalence Within Parties Over Time
    # ---------------------------------------------------------
    logger.info("Generating Plot 2: Topic Prevalence per Party...")
    # Filter to main parties to avoid clutter? Or plot all. 
    # Let's plot main parties (top 5-6 by volume) for clarity, or user might want all.
    # User asked for "different parties". Let's do all but maybe faceted.
    
    prevalence_party = df_merged.groupby(["year", "party", "topic_label"]).size().reset_index(name="count")
    party_year_totals = df_merged.groupby(["year", "party"]).size().reset_index(name="party_year_total")
    prevalence_party = prevalence_party.merge(party_year_totals, on=["year", "party"])
    prevalence_party["prevalence"] = prevalence_party["count"] / prevalence_party["party_year_total"]
    
    # Facet by Party (showing what topics they talk about)
    # Facet by Party (showing what topics they talk about)
    # User requested ALL parties.
    top_parties = df_merged["party"].unique().tolist()
    prevalence_party_filtered = prevalence_party # No filtering
    
    fig2 = px.line(
        prevalence_party_filtered,
        x="year",
        y="prevalence",
        color="topic_label",
        facet_col="party",
        facet_col_wrap=3,
        title="Topic Prevalence by Party (Top 9 Parties)",
        labels={"prevalence": "Share of Speeches"},
        height=1000
    )
    fig2.update_traces(mode='lines') # remove markers to reduce clutter
    fig2.write_html(output_dir / "02_topic_prevalence_by_party.html")

    # ---------------------------------------------------------
    # PLOT 3: Sentiment Distribution by Party within Topics
    # ---------------------------------------------------------
    logger.info("Generating Plot 3: Sentiment Distributions...")
    # Calculate sentiment
    logger.info("Loading pre-calculated sentiment scores...")
    sentiment_path = cfg.paths.exports_dir / "sentiment_scores.parquet"
    
    if sentiment_path.exists():
        df_sentiment = pd.read_parquet(sentiment_path)
        # Merge sentiment into df_merged
        df_scored = df_merged.merge(df_sentiment[["doc_id", "sentiment"]], on="doc_id", how="left")
        logger.info("Loaded sentiment scores from disk.")
    else:
        logger.warning("Sentiment scores not found on disk. Recalculating (this may be slow)...")
        provider = resolve_provider(cfg.sentiment)
        df_scored = score_documents(df_merged, "text", provider)
    
    # We want to see how sentiment differs for parties WITHIN specific topics.
    # Facet by Topic, X=Party, Y=Sentiment.
    
    # Filter to top parties again
    df_scored_filtered = df_scored[df_scored["party"].isin(top_parties)]
    
    fig3 = px.box(
        df_scored_filtered,
        x="party",
        y="sentiment",
        color="party",
        facet_col="topic_label",
        facet_col_wrap=3,
        title="Sentiment Distribution by Party per Topic",
        labels={"sentiment": "Sentiment Score (-1 to 1)"},
        height=1200
    )
    fig3.write_html(output_dir / "03_sentiment_by_party_topic.html")
    
    logger.info(f"Done! Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
