"""
visualizations.py - create plots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def prepare_analysis_data(df_sample, topics, topic_labels):
    """Prepare data. We remove the most generic topics (0) and the outliers (-1)"""
    df_sample["topic"] = topics
    
    # remove outliers (-1) and generic topic (0)
    df_analysis = df_sample[(df_sample["topic"] != -1) & (df_sample["topic"] != 0)].copy()
    
    print(f"\nAnalysis data prepared:")
    print(f"  Total speeches: {len(df_sample):,}")
    print(f"  Valid for analysis: {len(df_analysis):,}")
    print(f"  Topics: {df_analysis["topic"].nunique()}")
    
    return df_analysis


def plot_top_topics(df_analysis, topic_labels, output_path="/work/Exam/out/plots/top_topics_bar.png"):
    """Bar chart of most discussed topics"""
    topic_counts = df_analysis["topic"].value_counts().head(10)
    topic_names = [topic_labels.get(t, f"Topic {t}") for t in topic_counts.index]
    topic_names_short = [label[:50] + "..." if len(label) > 50 else label for label in topic_names]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(topic_counts)), topic_counts.values, 
                   color="steelblue", edgecolor="navy", alpha=0.6)
    
    plt.xlabel("Topics", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Speeches", fontsize=12)
    plt.title("Top 10 Topics in Folketinget (2020-2022)", fontsize=13, fontweight="bold")
    plt.xticks(range(len(topic_counts)), topic_names_short, rotation=45, ha="right")
    
    # add counts on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f"{int(height):,}", ha="center", va="bottom")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# create heatmap of what the various parties discusses
def plot_party_topic_heatmap(df_analysis, topic_labels, min_speeches=200, 
                             output_path="/work/Exam/out/plots/party_topic_heatmap.png"):
    """Heatmap with topic distribution across parties"""
    # Calculate percentages by party
    topic_party_counts = df_analysis.groupby(["party", "topic"]).size().unstack(fill_value=0)
    topic_party_props = topic_party_counts.div(topic_party_counts.sum(axis=1), axis=0) * 100
    
    # Filter to actual sig. parties
    party_speech_counts = df_analysis["party"].value_counts()
    significant_parties = party_speech_counts[party_speech_counts >= min_speeches].index
    topic_party_props_filtered = topic_party_props.loc[significant_parties]
    
    # Create labels
    readable_labels = {}
    for col in topic_party_props_filtered.columns:
        label = topic_labels.get(col, f"Topic {col}")
        readable_labels[col] = label[:35] + "..." if len(label) > 35 else label
    
    topic_party_props_filtered.columns = [readable_labels[col] for col in topic_party_props_filtered.columns]
    
    # plotting it
    plt.figure(figsize=(18, 10))
    sns.heatmap(topic_party_props_filtered, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={"label": "Percentage of Party Speeches (%)"}, 
                linewidths=0.5, linecolor="white")
    
    plt.title("Topic Distribution Across Political Parties (2020-2022)", 
              fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Topics", fontsize=12, fontweight="bold")
    plt.ylabel("Political Party", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_stacked_topics(df_analysis, topic_labels, min_speeches=200,
                       output_path="/work/Exam/out/plots/party_topic_stacked.png"):
    """Stacked bar chart of top 5 topics by party"""
    # get top 5 topics
    top_5_topics = df_analysis["topic"].value_counts().head(5).index
    df_top = df_analysis[df_analysis["topic"].isin(top_5_topics)]
    
    # Filter to sig. parties
    party_speech_counts = df_analysis["party"].value_counts()
    significant_parties = party_speech_counts[party_speech_counts >= min_speeches].index
    
    # Create percentages
    pivot_data = df_top.groupby(["party", "topic"]).size().unstack(fill_value=0)
    pivot_data = pivot_data.loc[significant_parties]
    pivot_data_pct = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
    
    # rename the columns
    readable_labels = {col: topic_labels.get(col, f"Topic {col}") for col in pivot_data_pct.columns}
    pivot_data_pct.columns = [readable_labels[col][:35] for col in pivot_data_pct.columns]
    
    # actual plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_data_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", 
                        width=0.75, edgecolor="black", linewidth=0.5)
    
    plt.title("Top 5 Topics Distribution by Party (2020-2022)", fontsize=14, fontweight="bold")
    plt.xlabel("Political Party", fontsize=12, fontweight="bold")
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def export_results(df_analysis, topic_model, topic_labels, topics, 
                  output_dir="/work/Exam/out/stats/"):
    
    # Topic details
    topic_info = topic_model.get_topic_info()
    topic_info_export = topic_info[topic_info["Topic"] != -1].copy()
    topic_info_export["Label"] = topic_info_export["Topic"].map(topic_labels)
    
    topic_words_detail = {}
    for topic_id in topic_info_export["Topic"]:
        words = topic_model.get_topic(topic_id)
        if words:
            topic_words_detail[topic_id] = ", ".join([f"{w[0]} ({w[1]:.3f})" for w in words[:10]])
    
    topic_info_export["Top_10_Words"] = topic_info_export["Topic"].map(topic_words_detail)
    topic_info_export[["Topic", "Label", "Count", "Top_10_Words"]].to_csv(
        f"{output_dir}/topic_details.csv", index=False
    )
    
    # Party-topic matrix
    party_topic_matrix = df_analysis.groupby(["party", "topic"]).size().unstack(fill_value=0)
    party_topic_matrix.columns = [topic_labels.get(col, f"Topic_{col}") for col in party_topic_matrix.columns]
    party_topic_matrix.to_csv(f"{output_dir}/party_topic_matrix.csv")
    
    # Summary
    topics_array = np.array(topics)
    with open(f"{output_dir}/summary_statistics.txt", "w", encoding="utf-8") as f:
        f.write("DANISH PARLIAMENT TOPIC ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write("METHODOLOGY\n")
        f.write("  Topic Model: BERTopic\n")
        f.write("  Stopwords: spaCy Danish + domain terms\n")
        f.write("  Representation: MaximalMarginalRelevance\n")
        f.write("  Time Period: 2020-2022\n\n")
        f.write(f"RESULTS\n")
        f.write(f"  Total speeches: {len(df_analysis):,}\n")
        f.write(f"  Valid topics: {df_analysis["topic"].nunique()}\n")
        f.write(f"  Outliers: {(topics_array == -1).sum():,}\n\n")
        f.write("TOP TOPICS\n")
        topic_dist = df_analysis["topic"].value_counts().head(10)
        for topic_id, count in topic_dist.items():
            label = topic_labels.get(topic_id, f"Topic {topic_id}")
            pct = (count / len(df_analysis)) * 100
            f.write(f"  {topic_id}. {label}: {count:,} ({pct:.1f}%)\n")


def create_all_visualizations(df_analysis, topic_model, topic_labels, topics):
    """Generate all visualizations and exports"""
    print("CREATING VISUALIZATIONS")
    
    plot_top_topics(df_analysis, topic_labels)
    plot_party_topic_heatmap(df_analysis, topic_labels)
    plot_stacked_topics(df_analysis, topic_labels)
    export_results(df_analysis, topic_model, topic_labels, topics)