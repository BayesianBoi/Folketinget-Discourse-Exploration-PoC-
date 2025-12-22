#!/usr/bin/env python3
"""
Compute topic coherence scores from existing pipeline outputs.
Uses gensim's CoherenceModel with c_v measure (correlates well with human judgments).

This script does NOT rerun the topic model - it uses the existing topics.json and preprocessed data.
"""

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.utils.logging import get_logger, setup_logging

def main():
    setup_logging()
    logger = get_logger(__name__)
    cfg = DEFAULT_CONFIG
    
    # Load preprocessed texts
    data_path = cfg.paths.processed_dir / "preprocessed.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"No preprocessed data at {data_path}")
    
    logger.info("Loading preprocessed data...")
    df = pd.read_parquet(data_path)
    texts = df["text"].fillna("").astype(str).tolist()
    
    # Load topics
    topics_path = cfg.paths.exports_dir / cfg.export.topic_json
    if not topics_path.exists():
        raise FileNotFoundError(f"No topics at {topics_path}")
    
    logger.info("Loading topics...")
    with open(topics_path, "r", encoding="utf-8") as f:
        topics_payload = json.load(f)
    
    topic_terms = topics_payload.get("topics", {})
    labels = topics_payload.get("metadata", {}).get("labels", {})
    
    # Tokenize texts (simple whitespace tokenization for coherence)
    logger.info("Tokenizing %d documents for coherence calculation...", len(texts))
    tokenized_texts = [text.lower().split() for text in texts]
    
    # Build gensim dictionary
    logger.info("Building dictionary...")
    dictionary = Dictionary(tokenized_texts)
    
    # Extract topic word lists
    topic_word_lists = []
    topic_ids = sorted([int(k) for k in topic_terms.keys()])
    
    for tid in topic_ids:
        terms = topic_terms[str(tid)]
        words = [t["term"] for t in terms if isinstance(t, dict) and "term" in t]
        topic_word_lists.append(words)
    
    logger.info("Computing c_v coherence for %d topics...", len(topic_word_lists))
    
    # Compute coherence using c_v measure
    coherence_model = CoherenceModel(
        topics=topic_word_lists,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    # Get per-topic coherence
    per_topic_coherence = coherence_model.get_coherence_per_topic()
    overall_coherence = coherence_model.get_coherence()
    
    # Print results
    print("\n" + "="*80)
    print("TOPIC COHERENCE SCORES (c_v)")
    print("="*80)
    print(f"\nOverall mean coherence: {overall_coherence:.4f}\n")
    print("-"*80)
    print(f"{'Topic ID':<10} {'Label':<50} {'Coherence':>10}")
    print("-"*80)
    
    results = []
    for i, (tid, coh) in enumerate(zip(topic_ids, per_topic_coherence)):
        label = labels.get(str(tid), f"Topic {tid}")
        # Truncate label for display
        display_label = label[:47] + "..." if len(label) > 50 else label
        print(f"{tid:<10} {display_label:<50} {coh:>10.4f}")
        results.append({
            "topic_id": tid,
            "label": label,
            "coherence_cv": round(coh, 4)
        })
    
    print("-"*80)
    print(f"{'Mean':<10} {'':<50} {overall_coherence:>10.4f}")
    print("="*80)
    
    # Find min/max
    coherences = [r["coherence_cv"] for r in results]
    min_coh = min(coherences)
    max_coh = max(coherences)
    min_topic = [r for r in results if r["coherence_cv"] == min_coh][0]
    max_topic = [r for r in results if r["coherence_cv"] == max_coh][0]
    
    print(f"\nLowest:  Topic {min_topic['topic_id']} ({min_topic['label'][:40]}...) = {min_coh:.4f}")
    print(f"Highest: Topic {max_topic['topic_id']} ({max_topic['label'][:40]}...) = {max_coh:.4f}")
    
    # Save to CSV
    output_path = cfg.paths.exports_dir / "coherence_scores.csv"
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["topic_id", "label", "coherence_cv"])
        writer.writeheader()
        writer.writerows(results)
    
    logger.info("Saved coherence scores to %s", output_path)
    
    print("\n" + "="*80)
    print("FOR THE PAPER (Section 4.3.5):")
    print("="*80)
    print(f"The mean c_v coherence score across our {len(topic_ids)} topics was {overall_coherence:.2f}.")
    print(f"Individual scores ranged from {min_coh:.2f} to {max_coh:.2f}.")
    print("="*80)


if __name__ == "__main__":
    main()
