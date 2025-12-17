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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.utils.logging import setup_logging, get_logger
# from dk_politics_topics.modeling.embeddings import load_embedding_model # Helper doesn't exist

st.set_page_config(page_title="Folketinget Discourse Explorer", layout="wide")

@st.cache_resource
def load_data():
    cfg = DEFAULT_CONFIG
    processed_path = cfg.paths.processed_dir / "preprocessed.parquet"
    doc_topics_path = cfg.paths.exports_dir / cfg.export.doc_topics_parquet
    topics_json_path = cfg.paths.exports_dir / cfg.export.topic_json
    sentiment_path = cfg.paths.exports_dir / "sentiment_scores.parquet"
    
    if not all(p.exists() for p in [processed_path, doc_topics_path, topics_json_path]):
        return None, None, None, None, None

    df_docs = pd.read_parquet(processed_path)
    df_topics = pd.read_parquet(doc_topics_path)
    topics_payload = json.loads(topics_json_path.read_text(encoding="utf-8"))
    
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
    
    return df, labels_map, topic_terms, cfg, abbr_to_name, df_docs, df_topics

@st.cache_resource
def get_semantic_search_model(cfg):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer(cfg.embedding.model_name)
    
    # Load cached embeddings from disk
    # Hash logic from embeddings.py - tricky to replicate exactly without importing hash_config
    # So we look for the most recent .npy file in embeddings_dir or try to match
    # For now, let's assume the user runs the pipeline which caches them.
    # A robust way is to re-embed if needed, but that's slow.
    # Let's try to find any .npy file in embeddings dir that matches the count?
    
    emb_files = list(cfg.paths.embeddings_dir.glob("*.npy"))
    embeddings = None
    if emb_files:
        # Load the largest one? Or the most recent?
        # Let's pick the most recent one
        latest_emb = max(emb_files, key=lambda p: p.stat().st_mtime)
        embeddings = np.load(latest_emb)
        
    return model, embeddings

def main():
    st.title("🇩🇰 Folketinget Discourse Explorer")
    
    df, labels, terms, cfg, party_map, df_docs, df_topics = load_data()
    
    if df is None:
        st.error("Data not found. Please run the pipeline first!")
        return

    tabs = st.tabs(["📊 Overview", "🏛️ Party Analysis", "🔎 Semantic Search", "📝 Topic Inspection"])

    # --- TAB 1: OVERVIEW ---
    with tabs[0]:
        st.header("Discourse Over Time")
        
        all_topics = sorted(df["topic_label"].unique())
        selected_topics = st.multiselect("Select Topics to Compare", all_topics, default=all_topics[:3])
        
        if selected_topics:
            agg = df[df["topic_label"].isin(selected_topics)].groupby(["time_bin", "topic_label"]).size().reset_index(name="count")
            totals = df.groupby("time_bin").size().reset_index(name="total")
            agg = agg.merge(totals, on="time_bin")
            agg["prevalence"] = agg["count"] / agg["total"]
            
            fig = px.line(agg, x="time_bin", y="prevalence", color="topic_label", markers=True, 
                          title="Topic Prevalence per Year")
            st.plotly_chart(fig, width="stretch")

            # FEATURE 1: Temporal Party Breakdown (Who owns the topic?)
            if len(selected_topics) == 1:
                target_topic = selected_topics[0]
                st.subheader(f"Who owns the debate on '{target_topic}'?")
                
                single_topic_df = df[df["topic_label"] == target_topic]
                party_time = single_topic_df.groupby(["time_bin", "party_name"]).size().reset_index(name="count")
                
                # Normalize to show share? Or raw count? Raw count shows volume.
                # Switching to Line Chart to allow direct comparison of peaks (Area chart stacks by default)
                fig_vol = px.line(party_time, x="time_bin", y="count", color="party_name",
                                   title=f"Volume of Speeches on '{target_topic}' by Party",
                                   line_group="party_name", markers=True)
                st.plotly_chart(fig_vol, width="stretch")

    # --- TAB 2: PARTY ANALYSIS ---
    with tabs[1]:
        st.header("Party Profiles")
        
        # Use full names for selection
        all_parties = sorted(df["party_name"].unique())
        sel_parties = st.multiselect("Select Parties", all_parties, default=all_parties[:min(4, len(all_parties))])
        
        if sel_parties:
            sub = df[df["party_name"].isin(sel_parties)]
            
            st.subheader("Who talks about what?")
            
            # Prepare data for Heatmap with Hover Sentiment
            # We need to aggregate prevalence AND mean sentiment for each (party, topic) pair
            heatmap_data = (
                sub.groupby(["topic_label", "party_name"])
                .agg(
                    count=("doc_id", "count"),
                    sentiment=("sentiment", "mean") if "sentiment" in sub.columns else ("doc_id", lambda x: 0)
                )
                .reset_index()
            )
            
            # Pivot for prevalence matrix (Z)
            z_data = heatmap_data.pivot(index="topic_label", columns="party_name", values="count").fillna(0)
            # Normalize by column (party) to show share of that party's speech
            z_data_norm = z_data.div(z_data.sum(axis=0), axis=1)
            
            # Pivot for sentiment matrix (Custom Data)
            s_data = heatmap_data.pivot(index="topic_label", columns="party_name", values="sentiment").fillna(0)
            
            # Explicitly align sentiment to probability matrix
            s_aligned = s_data.reindex(index=z_data_norm.index, columns=z_data_norm.columns).fillna(0)
            
            # Create a text matrix for reliable hover info
            # We pre-format the sentiment score. "N/A" if 0? No, just format everything.
            text_matrix = s_aligned.map(lambda x: f"{x:.2f}")

            # Use go.Heatmap for precise control over customdata
            fig_heat = go.Figure(data=go.Heatmap(
                z=z_data_norm.values,
                x=z_data_norm.columns,
                y=z_data_norm.index,
                colorscale="Blues",
                text=text_matrix.values,
                hovertemplate="<b>%{y}</b><br>%{x}<br>Share: %{z:.1%}<br>Avg Sentiment: %{text}<extra></extra>"
            ))
            
            fig_heat.update_layout(
                title="Topic Share per Party (Hover for Sentiment)",
                xaxis_title=None,
                yaxis_title=None
            )
            
            st.plotly_chart(fig_heat, width="stretch")

            # FEATURE 3: Sentiment Polarization
            st.divider()
            st.subheader("Polarization & Sentiment")
            pol_topic = st.selectbox("Select Topic to Analyze Sentiment Split", sorted(df["topic_label"].unique()))
            
            pol_df = df[df["topic_label"] == pol_topic]
            if "sentiment" in pol_df.columns:
                party_sent = pol_df.groupby("party_name")["sentiment"].mean().reset_index().sort_values("sentiment")
                
                # Diverging bar chart
                fig_pol = px.bar(party_sent, x="sentiment", y="party_name", orientation='h',
                                 color="sentiment", color_continuous_scale="RdBu",
                                 title=f"Sentiment Polarity on '{pol_topic}' (Red=Negative, Blue=Positive)",
                                 range_x=[-1, 1])
                st.plotly_chart(fig_pol, width="stretch")
            else:
                st.info("Sentiment data not available for polarization analysis.")

    # --- TAB 3: SEMANTIC SEARCH ---
    with tabs[2]:
        st.header("Find Speeches by Meaning")
        query = st.text_input("Enter a concept (e.g., 'klimakrise' or 'ældrepleje'):")
        
        if query:
            st.write("Encoding query...")
            model, embeddings = get_semantic_search_model(cfg)
            
            if embeddings is not None and len(embeddings) == len(df_docs):
                from sentence_transformers import util
                import torch
                import numpy as np
                
                query_vec = model.encode(query, convert_to_tensor=True)
                emb_tensor = torch.from_numpy(embeddings) if not isinstance(embeddings, torch.Tensor) else embeddings
                # Fix device mismatch
                emb_tensor = emb_tensor.to(query_vec.device)
                
                # Compute ALL cosine similarities
                # This returns a tensor of shape (1, n_docs)
                all_scores = util.cos_sim(query_vec, emb_tensor)[0].cpu().numpy()
                
                # Create a temporary DF for aggregation
                # We can use the fast index `df_docs` if it aligns with embeddings
                # We assume alignment (0-N index)
                
                # To aggregate by party, we need to join with `df` (which has party names)
                # This could be slow if we create a huge DF. 
                # Optimization: Filter top N results (e.g. top 1000) for UI, or use vectorized mapping if possible.
                # But user wants "find docs from ALL parties... give semantic score".
                # Aggregating 400k scores is fast. Mapping 400k doc_ids to parties is the bottleneck.
                
                # Strategy: Add score to `df_docs` momentarily? No, not thread safe.
                # Let's create a Series of scores indexed by doc_id (if df_docs has doc_id as column)
                
                # Optimization: Top 2000 hits is usually enough to gauge party stance
                top_k_agg = 5000 
                top_indices = np.argpartition(all_scores, -top_k_agg)[-top_k_agg:]
                
                # Extract data for top K
                top_scores = all_scores[top_indices]
                top_doc_ids = df_docs.iloc[top_indices]["doc_id"].values
                
                # Create mini dataframe
                res_df = pd.DataFrame({"doc_id": top_doc_ids, "score": top_scores})
                
                # Join with main metadata df (for party names)
                # df is indexed by doc_id, so join is fast
                res_df = res_df.join(df[["party_name", "time_bin", "topic_label", "text"]], on="doc_id", how="inner")
                
                # 1. Party Relevance Score (Mean of top K relevant speeches)
                # Or Max score? Mean is better for "alignment".
                party_agg = res_df.groupby("party_name")["score"].mean().reset_index().sort_values("score", ascending=False)
                
                st.subheader("Party Alignment with Query")
                party_agg = res_df.groupby("party_name")["score"].mean().reset_index().sort_values("score", ascending=False)
                
                fig_party = px.bar(party_agg, x="party_name", y="score", color="score", 
                                   title=f"Average Semantic Match (Top {top_k_agg} speeches)",
                                   range_y=[res_df["score"].min(), res_df["score"].max()])
                st.plotly_chart(fig_party, width="stretch")
                
                # 2. Top Matches from ALL parties (or top parties)
                st.subheader(f"Top Semantic Matches")
                
                # Sort descending
                res_df = res_df.sort_values("score", ascending=False)
                
                # Show top single hit
                best = res_df.iloc[0]
                st.markdown(f"**Best Match:** {best['party_name']} ({best['time_bin']}) - Score: {best['score']:.2f}")
                st.info(f"Topic: {best['topic_label']}\n\n\"{best['text']}\"")
                
                st.divider()
                st.markdown("### Explore Top Hits by Party")
                
                # Show top 2 hits for each party present in the top results
                unique_parties = res_df["party_name"].unique()
                for p in sorted(unique_parties):
                    p_hits = res_df[res_df["party_name"] == p].head(2)
                    with st.expander(f"See matches for {p}"):
                        for _, row in p_hits.iterrows():
                             st.markdown(f"**{row['time_bin']}** (Score: {row['score']:.2f})")
                             st.markdown(f"_{row['text']}_")
                             st.markdown("---")

            else:
                st.warning("⚠️ Could not align embeddings with document list.")
                # Fallback
                st.info("Tip: Run `python scripts/02_fit_topeax.py` again to ensure embeddings and data are synced.")
                
                # Fallback
                matches = df[df["text"].str.contains(query, case=False, na=False)].head(10)
                if not matches.empty:
                    for idx, row in matches.iterrows():
                         with st.expander(f"{row['party_name']} ({row['time_bin']})"):
                            st.write(f"**Topic**: {row['topic_label']}")
                            st.write(row['text'])

    # --- TAB 4: TOPIC INSPECTION ---
    with tabs[3]:
        st.header("Inspect Topic Content")
        target_topic_label = st.selectbox("Select Topic", sorted(df["topic_label"].unique()))
        
        target_tid = int(target_topic_label.split(":")[0])
        terms_list = terms.get(str(target_tid), [])
        
        st.subheader("Top Terms (Word Cloud)")
        
        # WordCloud
        if terms_list:
            # Prepare frequencies
            freqs = {}
            for t in terms_list:
                if isinstance(t, dict): 
                    freqs[t.get('term')] = t.get('weight', 0)
                elif isinstance(t, list):
                    freqs[t[0]] = t[1]
            
            if freqs:
                wc = WordCloud(width=600, height=300, background_color="white", colormap="viridis").generate_from_frequencies(freqs)
                st.image(wc.to_array(), width=600)
            else:
                st.write("No term weights available.")
        
        st.subheader("Most Representative Speeches")
        # Use probabilities from df_topics
        # Column name is just the integer ID
        topic_col = str(target_tid)
        
        if topic_col in df_topics.columns:
            # Get top 5 sorted by probability
            top_indices = df_topics[topic_col].nlargest(5).index
            # Retrieve from main df (which is indexed by doc_id)
            # Ensure intersection
            valid_indices = [idx for idx in top_indices if idx in df.index]
            subset = df.loc[valid_indices]
            
            # Add probability to display
            # We locate probabilities again - safe since indices match
            subset["prob"] = df_topics.loc[valid_indices, topic_col]
            
            for idx, row in subset.iterrows():
                st.markdown(f"**{row['party']} ({row['time_bin']})** (Confidence: {row['prob']:.1%})")
                st.markdown(f"> {row['text']}")
                st.divider()
        else:
            st.warning(f"Probability scores for topic {target_tid} not found. Showing random sample.")
            subset = df[df["top_topic_id"] == target_tid].sample(min(5, len(df)))
            for idx, row in subset.iterrows():
                st.markdown(f"**{row['party']} ({row['time_bin']})**: {row['text']}")
                st.divider()

if __name__ == "__main__":
    main()
