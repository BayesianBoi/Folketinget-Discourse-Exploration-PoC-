import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
src_path = str(SRC)
if src_path in sys.path:
    sys.path.remove(src_path)
sys.path.insert(0, src_path)

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.utils.logging import setup_logging, get_logger
try:
    from dashboard.agent import AIAgent
except ImportError:
    from agent import AIAgent
# from dk_politics_topics.modeling.embeddings import load_embedding_model # Helper doesn't exist

# Logo path
LOGO_PATH = Path(__file__).parent / "logo_circle.png"

st.set_page_config(
    page_title="Folketinget Discourse Explorer", 
    layout="wide",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🇩🇰"
)

# Hide Streamlit settings/footer
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    cfg = DEFAULT_CONFIG
    processed_path = cfg.paths.processed_dir / "preprocessed.parquet"
    doc_topics_path = cfg.paths.exports_dir / cfg.export.doc_topics_parquet
    topics_json_path = cfg.paths.exports_dir / cfg.export.topic_json
    sentiment_path = cfg.paths.exports_dir / "sentiment_scores.parquet"
    metadata_path = cfg.paths.exports_dir / cfg.export.metadata_json
    
    if not all(p.exists() for p in [processed_path, doc_topics_path, topics_json_path]):
        return None, None, None, None, None, None, None, None

    df_docs = pd.read_parquet(processed_path)
    df_topics = pd.read_parquet(doc_topics_path)
    topics_payload = json.loads(topics_json_path.read_text(encoding="utf-8"))
    metadata = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = None
    
    # Process Labels
    labels_map = topics_payload.get("metadata", {}).get("labels", {})
    topic_terms = topics_payload.get("topics", {})
    
    # Merge
    topic_cols = [c for c in df_topics.columns if c != "doc_id"]
    df_topics["top_topic_id"] = df_topics[topic_cols].idxmax(axis=1).apply(
        lambda x: int(x.split("_")[-1]) if isinstance(x, str) else int(x)
    )
    
    df = df_docs.merge(df_topics[["doc_id", "top_topic_id"]], on="doc_id", how="inner")
    
    # Load Sentiment if available
    if sentiment_path.exists():
        df_sent = pd.read_parquet(sentiment_path)
        df = df.merge(df_sent[["doc_id", "sentiment"]], on="doc_id", how="left")
    
    # Filter Ignored Topics
    if cfg.topeax.ignored_topics:
        df = df[~df["top_topic_id"].isin(cfg.topeax.ignored_topics)]
    
    # Enrich with labels
    def get_fmt_label(tid):
        return f"{tid}: {labels_map.get(str(tid), f'Topic {tid}')}"
    
    df["topic_label"] = df["top_topic_id"].apply(get_fmt_label)
    
    # Map Party Names (Inverse mapping from config)
    # config has: "SOCIALDEMOKRATIET": "S"
    # we want: "S": "Socialdemokratiet"
    abbr_to_name = {v: k.title() for k, v in cfg.corpus.party_normalization.items()}
    df["party_name"] = df["party"].map(abbr_to_name).fillna(df["party"])
    
    # Set index for fast lookups
    df.set_index("doc_id", inplace=True, drop=False)
    df_topics.set_index("doc_id", inplace=True) 
    
    return df, labels_map, topic_terms, cfg, abbr_to_name, df_docs, df_topics, metadata

@st.cache_resource
def get_semantic_search_model(cfg, n_docs: int):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from dk_politics_topics.utils.cache import hash_config, maybe_load_numpy
    
    model = SentenceTransformer(cfg.embedding.model_name)
    
    doc_prefix = "passage" if "e5" in (cfg.embedding.model_name or "").lower() else ""
    expected_cache_name = cfg.embedding.cache_name or hash_config(
        {"model": cfg.embedding.model_name, "n": int(n_docs), "doc_prefix": doc_prefix}
    )
    expected_path = cfg.paths.embeddings_dir / f"{expected_cache_name}.npy"

    # Preferred: load the exact cache file produced by `embed_texts(...)`.
    embeddings = maybe_load_numpy(cfg.paths.embeddings_dir, expected_cache_name, mmap_mode="c")

    loaded_path = expected_path if embeddings is not None else None
    found_files = []

    # Fallback: pick any .npy that matches row count (useful if multiple caches exist).
    if embeddings is None and cfg.paths.embeddings_dir.exists():
        for p in sorted(cfg.paths.embeddings_dir.glob("*.npy")):
            try:
                arr = np.load(p, mmap_mode="c")
                found_files.append({"path": str(p), "shape": tuple(arr.shape), "dtype": str(arr.dtype)})
                if len(arr.shape) == 2 and int(arr.shape[0]) == int(n_docs):
                    embeddings = arr
                    loaded_path = p
                    break
            except Exception:
                continue

    info = {
        "expected_cache_name": expected_cache_name,
        "expected_path": str(expected_path),
        "loaded_path": str(loaded_path) if loaded_path is not None else None,
        "found_files": found_files,
    }

    return model, embeddings, info

    return model, embeddings, info

def _get_topic_display_name(full_label: str) -> str:
    """Extract just the topic name without the number prefix (e.g., '5: Climate' -> 'Climate')."""
    if ":" in full_label:
        return full_label.split(":", 1)[1].strip()
    return full_label

def _sort_topics_alphabetically(topic_labels: list) -> list:
    """Sort topic labels alphabetically by their display name (ignoring the number prefix)."""
    return sorted(topic_labels, key=lambda x: _get_topic_display_name(x).lower())

def main():
    col1, col2 = st.columns([1, 20], vertical_alignment="bottom")
    with col1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=80)
    with col2:
        st.title("Folketinget Discourse Explorer")
    
    df, labels, terms, cfg, party_map, df_docs, df_topics, metadata = load_data()
    
    if df is None:
        st.error("Data not found. Please run the pipeline first!")
        return

    tabs = st.tabs(["📊 Overview", "🔎 Semantic Search", "📝 Topic Inspection", "🏛️ Party Analysis", "🤖 AI Analyst (BETA)"])

    # --- TAB 1: OVERVIEW ---
    with tabs[0]:
        st.header("Discourse Over Time")

        all_topics = _sort_topics_alphabetically(df["topic_label"].unique().tolist())
        # Default to "Minkaflivning under COVID" if it exists
        default_topic = [t for t in all_topics if "minkaflivning" in t.lower()][:1] or all_topics[:1]
        selected_topics = st.multiselect(
            "Select Topics to Compare", 
            all_topics, 
            default=default_topic,
            format_func=_get_topic_display_name
        )

        if selected_topics:
            agg = (
                df[df["topic_label"].isin(selected_topics)]
                .groupby(["time_bin", "topic_label"])
                .size()
                .reset_index(name="count")
            )
            totals = df.groupby("time_bin").size().reset_index(name="total")
            agg = agg.merge(totals, on="time_bin")
            agg["prevalence"] = agg["count"] / agg["total"]

            fig = px.line(
                agg,
                x="time_bin",
                y="prevalence",
                color="topic_label",
                markers=True,
                title="Topic Prevalence per Year",
            )
            st.plotly_chart(fig, width="stretch")

            # FEATURE 1: Temporal Party Breakdown (Who owns the topic?)
            if len(selected_topics) == 1:
                target_topic = selected_topics[0]
                target_display = _get_topic_display_name(target_topic)
                st.subheader(f"Who owns the debate on '{target_display}'?")

                single_topic_df = df[df["topic_label"] == target_topic]
                party_time = (
                    single_topic_df.groupby(["time_bin", "party_name"])
                    .size()
                    .reset_index(name="count")
                )

                # Create complete time series with all years and all parties
                all_years = sorted(df["time_bin"].unique())
                all_parties_in_topic = party_time["party_name"].unique().tolist()
                
                # Create a complete index of all year-party combinations
                from itertools import product
                complete_index = pd.DataFrame(
                    list(product(all_years, all_parties_in_topic)),
                    columns=["time_bin", "party_name"]
                )
                
                # Merge with actual data and fill missing with 0
                party_time = complete_index.merge(
                    party_time, on=["time_bin", "party_name"], how="left"
                )
                party_time["count"] = party_time["count"].fillna(0).astype(int)

                fig_vol = px.line(
                    party_time,
                    x="time_bin",
                    y="count",
                    color="party_name",
                    title=f"Volume of Speeches on '{target_display}' by Party",
                    line_group="party_name",
                    markers=True,
                )
                st.plotly_chart(fig_vol, width="stretch")

        st.divider()
        with st.expander("Data & model card"):
            generated_at = metadata.get("generated_at") if isinstance(metadata, dict) else None

            years = pd.to_numeric(df_docs.get("year"), errors="coerce")
            year_min = int(years.min()) if not years.isna().all() else None
            year_max = int(years.max()) if not years.isna().all() else None

            corpus_parties = None
            if isinstance(metadata, dict):
                corpus_parties = (metadata.get("corpus") or {}).get("parties")
            if not corpus_parties:
                corpus_parties = sorted(df_docs["party"].dropna().unique().tolist())

            st.markdown("**Dataset**")
            st.write(
                {
                    "generated_at": generated_at,
                    "speeches": int(len(df_docs)),
                    "years": f"{year_min}–{year_max}" if year_min and year_max else None,
                    "parties_included": len(corpus_parties),
                    "min_party_share_filter": getattr(cfg.corpus, "min_party_share", None),
                }
            )
            st.markdown("**Models**")
            st.write(
                {
                    "embedding_model": cfg.embedding.model_name,
                    "e5_prefix_formatting": bool("e5" in (cfg.embedding.model_name or "").lower()),
                    "topic_model": "Topeax",
                    "topic_count": int(len(terms)) if isinstance(terms, dict) else None,
                    "sentiment_approach": cfg.sentiment.approach,
                    "sentiment_model": cfg.sentiment.huggingface_model,
                }
            )
            st.markdown("**Parties**")
            st.caption(", ".join(corpus_parties))

    # --- TAB 4: PARTY ANALYSIS ---
    with tabs[3]:
        st.header("Party Profiles")

        all_parties = sorted(df["party_name"].unique())
        sel_parties = st.multiselect(
            "Select Parties",
            all_parties,
            default=all_parties,  # Default to all parties
        )

        if sel_parties:
            sub = df[df["party_name"].isin(sel_parties)]

            st.subheader("Who talks about what?")
            
            # Let user select which topics to show
            all_topics_for_filter = _sort_topics_alphabetically(sub["topic_label"].unique().tolist())
            selected_topics_for_heatmap = st.multiselect(
                "Select topics to display (leave empty for all)",
                all_topics_for_filter,
                default=[],
                format_func=_get_topic_display_name,
                help="Filter which topics appear in the heatmap. Leave empty to show all."
            )
            
            # Apply topic filter if any selected
            if selected_topics_for_heatmap:
                sub_filtered = sub[sub["topic_label"].isin(selected_topics_for_heatmap)]
            else:
                sub_filtered = sub

            heatmap_data = (
                sub_filtered.groupby(["topic_label", "party_name"])
                .agg(
                    count=("doc_id", "count"),
                    sentiment=("sentiment", "mean")
                    if "sentiment" in sub_filtered.columns
                    else ("doc_id", lambda x: 0),
                )
                .reset_index()
            )
            
            # Add display names for y-axis
            heatmap_data["topic_display"] = heatmap_data["topic_label"].apply(_get_topic_display_name)

            z_data = heatmap_data.pivot(index="topic_display", columns="party_name", values="count").fillna(0)
            z_data_norm = z_data.div(z_data.sum(axis=0), axis=1)
            
            # Calculate party totals for the calculation breakdown
            party_totals = z_data.sum(axis=0)
            
            # Use sentiment for color instead of count share
            s_data = heatmap_data.pivot(index="topic_display", columns="party_name", values="sentiment").fillna(0)
            s_aligned = s_data.reindex(index=z_data_norm.index, columns=z_data_norm.columns).fillna(0)
            
            # Create text matrix showing share with calculation: "15.2% (152/1000)"
            def format_share_with_calc(row_idx, col_idx):
                count = int(z_data.iloc[row_idx, col_idx])
                total = int(party_totals.iloc[col_idx])
                pct = z_data_norm.iloc[row_idx, col_idx]
                return f"{pct:.1%} ({count:,}/{total:,})"
            
            share_matrix = pd.DataFrame(
                [[format_share_with_calc(i, j) for j in range(len(z_data.columns))] for i in range(len(z_data.index))],
                index=z_data.index,
                columns=z_data.columns
            )
            
            # Custom Red-White-Green colorscale for sentiment
            sentiment_colorscale = [
                [0.0, "rgb(215, 48, 39)"],      # Red for -1
                [0.25, "rgb(252, 141, 89)"],    # Light red for -0.5
                [0.5, "rgb(255, 255, 255)"],    # White for 0
                [0.75, "rgb(145, 207, 96)"],    # Light green for 0.5
                [1.0, "rgb(26, 152, 80)"],      # Green for 1
            ]

            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=s_aligned.values,  # Color by sentiment
                    x=s_aligned.columns,
                    y=s_aligned.index,
                    colorscale=sentiment_colorscale,
                    zmin=-1,  # Force scale to -1
                    zmax=1,   # Force scale to 1
                    text=share_matrix.values,  # Pre-formatted share with calculation
                    hovertemplate="<b>%{y}</b><br>%{x}<br>Share: %{text}<br>Avg Sentiment: %{z:.2f}<extra></extra>",
                    colorbar=dict(
                        title="Sentiment",
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=["-1 (Negative)", "-0.5", "0 (Neutral)", "0.5", "1 (Positive)"],
                    ),
                )
            )
            fig_heat.update_layout(
                title="Topic Sentiment by Party (Red=Negative, White=Neutral, Green=Positive)",
                xaxis_title=None,
                yaxis_title=None,
                height=max(400, len(s_aligned) * 25),  # Dynamic height based on topics
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            st.divider()
            st.subheader("Polarization & Sentiment")
            all_topics_sorted = _sort_topics_alphabetically(df["topic_label"].unique().tolist())
            pol_topic = st.selectbox(
                "Select Topic to Analyze Sentiment Split",
                all_topics_sorted,
                format_func=_get_topic_display_name,
            )

            pol_df = df[df["topic_label"] == pol_topic]
            pol_topic_display = _get_topic_display_name(pol_topic)
            if "sentiment" in pol_df.columns:
                party_sent = (
                    pol_df.groupby("party_name")["sentiment"]
                    .mean()
                    .reset_index()
                    .sort_values("sentiment")
                )
                
                # Custom Red-White-Green colorscale for sentiment
                sentiment_colorscale = [
                    [0.0, "rgb(215, 48, 39)"],      # Red for -1
                    [0.25, "rgb(252, 141, 89)"],    # Light red for -0.5
                    [0.5, "rgb(255, 255, 255)"],    # White for 0
                    [0.75, "rgb(145, 207, 96)"],    # Light green for 0.5
                    [1.0, "rgb(26, 152, 80)"],      # Green for 1
                ]
                
                fig_pol = px.bar(
                    party_sent,
                    x="sentiment",
                    y="party_name",
                    orientation="h",
                    color="sentiment",
                    color_continuous_scale=sentiment_colorscale,
                    range_color=[-1, 1],  # Force color range to -1 to 1
                    title=f"Sentiment on '{pol_topic_display}' (Red=Negative, White=Neutral, Green=Positive)",
                    range_x=[-1, 1],
                )
                fig_pol.update_layout(
                    coloraxis_colorbar=dict(
                        title="Sentiment",
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=["-1", "-0.5", "0", "0.5", "1"],
                    ),
                )
                st.plotly_chart(fig_pol, use_container_width=True)
            else:
                st.info("Sentiment data not available for polarization analysis.")

    # --- TAB 2: SEMANTIC SEARCH ---
    with tabs[1]:
        st.header("Semantic Search")
        st.caption("Find speeches by meaning (relevance). This does not measure support/opposition.")
        st.info("Tip: Always read the quotes. High relevance ≠ agreement.")

        with st.form("search_form"):
            query_a = st.text_input("Query", placeholder="e.g., skattelettelser")
            submitted = st.form_submit_button("Search")

        def render_semantic_search() -> None:
            if not submitted:
                return

            if not query_a:
                st.info("Enter a query to run semantic search.")
                return

            model, embeddings, emb_info = get_semantic_search_model(cfg, n_docs=len(df_docs))

            if embeddings is None or int(getattr(embeddings, "shape", [0])[0]) != int(len(df_docs)):
                st.warning("⚠️ Could not align embeddings with document list.")
                loaded = emb_info.get("loaded_path") or "None"
                found = emb_info.get("found_files") or []
                found_summary = ", ".join(
                    f"{Path(f['path']).name} shape={f.get('shape')}"
                    for f in found[:3]
                    if isinstance(f, dict) and f.get("path")
                )
                if len(found) > 3:
                    found_summary += f", … (+{len(found) - 3} more)"
                st.caption(
                    "Embeddings cache mismatch: "
                    f"expected `{Path(emb_info['expected_path']).name}` for n={len(df_docs)}; "
                    f"loaded `{Path(loaded).name if loaded != 'None' else loaded}`; "
                    f"found [{found_summary or 'no .npy files found'}]."
                )
                st.info("Tip: Run `python scripts/02_fit_topeax.py` again to ensure embeddings and data are synced.")

                matches = df[df["text"].str.contains(query_a, case=False, na=False)].head(10)
                if not matches.empty:
                    st.subheader("Fallback: keyword matches")
                    for _, row in matches.iterrows():
                        with st.expander(f"{row['party_name']} ({row['time_bin']})"):
                            st.write(f"**Topic**: {row['topic_label']}")
                            st.write(row["text"])
                return

            from sentence_transformers import util
            import torch

            years_source = df_docs["year"] if "year" in df_docs.columns else df_docs.get("time_bin")
            years = pd.to_numeric(years_source, errors="coerce")
            year_min = int(years.min()) if not years.isna().all() else 2005
            year_max = int(years.max()) if not years.isna().all() else 2025

            year_start, year_end = st.slider(
                "Year range",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                help="Limits search and party baseline to speeches within the selected years.",
            )

            df_slice = df[df["year"].between(year_start, year_end, inclusive="both")].copy()
            if df_slice.empty:
                st.warning("No speeches found in the selected year range.")
                return

            with st.expander("Filters (optional)"):
                topics_available = _sort_topics_alphabetically(df_slice["topic_label"].unique().tolist())
                selected_topics = st.multiselect(
                    "Restrict to topics", 
                    topics_available, 
                    default=[],
                    format_func=_get_topic_display_name
                )

                parties_available = sorted(df_slice["party_name"].unique())
                focus_parties = st.multiselect(
                    "Show parties (display only)",
                    parties_available,
                    default=[],
                    help="Hides other parties in charts/tables for readability (does not change the baseline).",
                )

            if selected_topics:
                df_slice = df_slice[df_slice["topic_label"].isin(selected_topics)].copy()
                if df_slice.empty:
                    st.warning("No speeches found for the selected topics/year range.")
                    return

            allowed_ids = pd.Index(df_slice["doc_id"].unique())
            candidate_mask = df_docs["doc_id"].isin(allowed_ids).to_numpy()
            candidate_idx = np.flatnonzero(candidate_mask)
            if candidate_idx.size == 0:
                st.warning("No speeches found after applying filters.")
                return

            with st.spinner("Searching millions of words..."):
                model, embeddings, emb_info = get_semantic_search_model(cfg, n_docs=len(df_docs))

                if embeddings is None or int(getattr(embeddings, "shape", [0])[0]) != int(len(df_docs)):
                    st.warning("⚠️ Could not align embeddings with document list.")
                    # Fallback logic could go here, but for now we return
                    return

                df_docs_slice = df_docs.iloc[candidate_idx]
                emb_tensor = torch.from_numpy(embeddings) if not isinstance(embeddings, torch.Tensor) else embeddings

                query_text = query_a.strip()
                query_for_embedding = query_text
                if "e5" in (cfg.embedding.model_name or "").lower():
                    query_for_embedding = f"query: {query_text}"

                query_vecs = model.encode([query_for_embedding], convert_to_tensor=True)

                idx_t = torch.as_tensor(candidate_idx, dtype=torch.long, device=query_vecs.device)
                emb_slice = emb_tensor.to(query_vecs.device).index_select(0, idx_t)
                score_mat = util.cos_sim(query_vecs, emb_slice).cpu().numpy()

            def _compute_outputs(scores: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
                scores = np.asarray(scores, dtype=np.float64)
                n_scores = int(scores.shape[0])

                q50, q90, q99 = (float(x) for x in np.quantile(scores, [0.50, 0.90, 0.99]))

                lo = float(scores.min())
                hi = float(scores.max())
                if hi - lo <= 1e-12:
                    edges = np.array([lo - 1e-6, hi + 1e-6], dtype=np.float64)
                else:
                    edges = np.linspace(lo, hi, 501, dtype=np.float64)
                hist, edges = np.histogram(scores, bins=edges)
                cdf = np.cumsum(hist) / float(n_scores)

                top_k = min(2000, n_scores)
                top_idx = np.argpartition(scores, -top_k)[-top_k:]

                top_doc_ids = df_docs_slice.iloc[top_idx]["doc_id"].values
                top_scores = scores[top_idx]

                res = pd.DataFrame({"doc_id": top_doc_ids, "score": top_scores})
                score_bins = np.searchsorted(edges, res["score"].to_numpy(), side="right") - 1
                score_bins = np.clip(score_bins, 0, len(cdf) - 1)
                res["pct"] = cdf[score_bins]

                res = (
                    res.join(
                        df_slice[["party_name", "year", "time_bin", "topic_label", "text"]],
                        on="doc_id",
                        how="inner",
                    )
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )

                top_k_used = int(len(res))
                denom_top = max(1, top_k_used)

                corpus_counts = df_slice["party_name"].value_counts(dropna=False)
                party = pd.DataFrame({"party_name": corpus_counts.index, "count_corpus": corpus_counts.values})

                hits = (
                    res.groupby("party_name")
                    .agg(score=("score", "mean"), count_topk=("doc_id", "count"))
                    .reset_index()
                )
                party = party.merge(hits, on="party_name", how="left")
                party["count_topk"] = party["count_topk"].fillna(0).astype(int)

                denom_corpus = max(1, int(len(df_slice)))
                party["corpus_share"] = party["count_corpus"] / float(denom_corpus)
                party["hit_share_raw"] = party["count_topk"] / float(denom_top)

                alpha = 5.0
                n_parties = max(1, int(df_slice["party_name"].nunique()))
                party["hit_share"] = (party["count_topk"] + alpha) / float(denom_top + alpha * n_parties)
                party["times_expected"] = party["hit_share"] / party["corpus_share"].replace(0, np.nan)
                party["over_under"] = np.log2(party["times_expected"])

                party = party.sort_values("over_under", ascending=False).reset_index(drop=True)

                stats = {"n_scores": n_scores, "q50": q50, "q90": q90, "q99": q99, "top_k_used": top_k_used}
                return res, party, stats

            res_df, party_df, stats = _compute_outputs(score_mat[0])

            with st.expander("How to interpret this (read first)"):
                st.markdown(
                    "- **Party chart**: compares each party’s share among the top semantic matches to its share in the filtered corpus.\n"
                    "- **Over/Under = 0**: exactly as expected.\n"
                    "- **Over/Under = +1**: ~2× more present than expected.\n"
                    "- **Over/Under = −1**: ~½× as present as expected.\n"
                    "- This is about **relevance**, not agreement. Use the quotes below as evidence.\n"
                    "- Year/topic filters change the baseline (what “expected” means)."
                )

            def _party_chart(party_df: pd.DataFrame, title: str):
                if focus_parties:
                    party_df = party_df[party_df["party_name"].isin(focus_parties)].copy()
                abs_cap = (
                    float(np.nanquantile(np.abs(party_df["over_under"].to_numpy()), 0.95))
                    if len(party_df)
                    else 1.0
                )
                abs_cap = max(abs_cap, 0.25)
                fig = px.bar(
                    party_df,
                    x="over_under",
                    y="party_name",
                    orientation="h",
                    color="over_under",
                    color_continuous_scale=[(0.0, "#2166ac"), (0.5, "#f7f7f7"), (1.0, "#b2182b")],
                    range_color=[-abs_cap, abs_cap],
                    title=title,
                    hover_data={
                        "count_topk": True,
                        "hit_share_raw": ":.2%",
                        "count_corpus": True,
                        "corpus_share": ":.2%",
                        "times_expected": ":.2f",
                        "score": ":.3f",
                    },
                    labels={
                        "party_name": "Party",
                        "over_under": "Over/Under",
                        "count_topk": "Hits in top matches",
                        "hit_share_raw": "Share in top matches",
                        "count_corpus": "Total speeches (corpus)",
                        "corpus_share": "Share in corpus",
                        "times_expected": "Times expected (×)",
                        "score": "Avg similarity (cosine)",
                    },
                )
                fig.update_layout(
                    xaxis_title="Over/Under (0 = expected, +1 = 2×, −1 = ½×)",
                    yaxis_title=None,
                    coloraxis_colorbar=dict(title="Over/Under"),
                    xaxis=dict(range=[-abs_cap, abs_cap], zeroline=True, zerolinecolor="rgba(0,0,0,0.35)"),
                )
                fig.add_vline(x=0, line_width=2, line_dash="dot", line_color="rgba(0,0,0,0.35)")
                return fig

            st.subheader("Party Over/Under-Representation")
            st.caption(
                f"Searching {year_start}–{year_end} with {len(df_slice):,} speeches "
                f"(top {stats['top_k_used']:,} matches)."
            )

            st.plotly_chart(_party_chart(party_df, f"Query: {query_text}"), width="stretch")

            st.subheader("Evidence (read the best matches)")
            if focus_parties:
                res_df = res_df[res_df["party_name"].isin(focus_parties)].copy()
                party_df = party_df[party_df["party_name"].isin(focus_parties)].copy()

            best = res_df.iloc[0] if len(res_df) else None
            if best is not None:
                st.markdown(
                    f"**Best match:** {best['party_name']} ({int(best['year'])}) — "
                    f"pct={best['pct']:.2%}, score={best['score']:.2f}"
                )
                st.info(f"Topic: {best['topic_label']}\n\n\"{best['text']}\"")

            export_hits = res_df.sort_values("score", ascending=False).head(200)
            export_hits = export_hits[
                ["doc_id", "party_name", "year", "time_bin", "topic_label", "pct", "score", "text"]
            ]
            export_csv = export_hits.to_csv(index=False).encode("utf-8-sig")
            safe_name = "".join(c for c in query_text if c.isalnum() or c in (" ", "_", "-")).strip()
            safe_name = safe_name.replace(" ", "_")[:40] or "query"

            st.download_button(
                "Download top matches (CSV)",
                data=export_csv,
                file_name=f"semantic_matches_{safe_name}_{year_start}-{year_end}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            party_export = party_df[
                [
                    "party_name",
                    "count_topk",
                    "hit_share_raw",
                    "count_corpus",
                    "corpus_share",
                    "times_expected",
                    "over_under",
                ]
            ].copy()
            party_export_csv = party_export.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download party summary (CSV)",
                data=party_export_csv,
                file_name=f"semantic_party_{safe_name}_{year_start}-{year_end}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.subheader("Top matches (preview)")
            preview = export_hits.copy()
            preview["text_preview"] = preview["text"].astype(str).str.slice(0, 220)
            st.dataframe(
                preview.drop(columns=["text"]),
                use_container_width=True,
                hide_index=True,
            )

        render_semantic_search()

    # --- TAB 3: TOPIC INSPECTION ---
    with tabs[2]:
        st.header("Inspect Topic Content")
        all_topics_sorted = _sort_topics_alphabetically(df["topic_label"].unique().tolist())
        target_topic_label = st.selectbox(
            "Select Topic", 
            all_topics_sorted,
            format_func=_get_topic_display_name
        )

        target_tid = int(target_topic_label.split(":")[0])
        terms_list = terms.get(str(target_tid), [])
        
        # Show total speech count for this topic
        speech_count = len(df[df["top_topic_id"] == target_tid])
        st.metric("Total Speeches", f"{speech_count:,}")

        # Download button for topic data
        topic_speeches = df[df["top_topic_id"] == target_tid].copy()
        # Ensure we export useful columns
        export_cols = [c for c in ["doc_id", "year", "date", "party", "text"] if c in topic_speeches.columns]
        csv_data = topic_speeches[export_cols].to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            label="Download Speeches (.csv)",
            data=csv_data,
            file_name=f"topic_{target_tid}_speeches.csv",
            mime="text/csv",
        )

        st.subheader("Top Terms (Word Cloud)")

        if terms_list:
            freqs = {}
            for t in terms_list:
                if isinstance(t, dict):
                    freqs[t.get("term")] = t.get("weight", 0)
                elif isinstance(t, list) and len(t) >= 2:
                    freqs[t[0]] = t[1]

            if freqs:
                wc = WordCloud(
                    width=600,
                    height=300,
                    background_color="white",
                    colormap="viridis",
                ).generate_from_frequencies(freqs)
                st.image(wc.to_array(), width=600)
            else:
                st.write("No term weights available.")

        st.subheader("Most Representative Speeches")
        topic_col = str(target_tid)

        if topic_col in df_topics.columns:
            top_indices = df_topics[topic_col].nlargest(5).index
            valid_indices = [idx for idx in top_indices if idx in df.index]
            subset = df.loc[valid_indices].copy()
            subset["prob"] = df_topics.loc[valid_indices, topic_col]

            for _, row in subset.iterrows():
                st.markdown(f"**{row['party']} ({row['time_bin']})** (Confidence: {row['prob']:.1%})")
                st.markdown(f"> {row['text']}")
                st.divider()
        else:
            st.warning(f"Probability scores for topic {target_tid} not found. Showing random sample.")
            subset = df[df["top_topic_id"] == target_tid].sample(min(5, len(df)))
            for _, row in subset.iterrows():
                st.markdown(f"**{row['party']} ({row['time_bin']})**: {row['text']}")
                st.divider()

    # --- TAB 5: AI ANALYST ---
    with tabs[4]:
        st.header("Ask the AI Political Analyst")
        st.info("This is a writing/thinking aid. It may be wrong—verify claims by checking the underlying speeches and charts.")

        ss_model, ss_embeddings, _ = get_semantic_search_model(cfg, n_docs=len(df_docs))

        if "agent" not in st.session_state:
            st.session_state.agent = AIAgent(df, df_docs, terms, ss_model, ss_embeddings)
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if len(st.session_state.messages) == 0:
            st.markdown("### Try asking:")
            cols = st.columns(2)
            with cols[0]:
                st.info("What is the 'Klima' topic about?")
                st.info("How does Venstre feel about 'Skattepolitik'?")
            with cols[1]:
                st.info("When was 'Udlændinge' most discussed?")
                st.info("Compare Socialdemokratiet and SF on 'Velfærd'")

        if prompt := st.chat_input("Ask about topics, parties, or trends..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Analyzing with AI..."):
                response = st.session_state.agent.answer(prompt)

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # --- FOOTER ---
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.9em;">
                <p><strong>Folketinget Discourse Explorer</strong></p>
                <p>Created by <strong>Niels Værbak</strong> & <strong>Søren Meiner</strong></p>
                <p style="font-size: 0.85em;">
                    ⚠️ <em>This tool is for exploratory research purposes. 
                    Results should be verified against primary sources before being used in academic or professional contexts.
                    The AI features may produce inaccurate information.</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
