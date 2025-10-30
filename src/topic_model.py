"""
topic_model.py - Train BERTopic model
"""
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
import time

def train_bertopic(documents, stopwords, nr_topics=10, min_topic_size=25):
    """Train BERTopic model on documents."""
    print(f"\nTraining BERTopic on {len(documents):,} documents...")
    print(f"Config: {nr_topics} topics, min_size={min_topic_size}, {len(stopwords)} stopwords")
    
    # embedding model here (very good for danish, but we miight consider machine translation for the actual examerino)
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # setting up the countvec
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2), # moving with ngram of 1, 2
        stop_words=stopwords,
        min_df=2, # require a minimum of 2 documents
        max_df=0.95, # taking a maximum of 95% of the documents
        max_features=5000 # maximum features of 5k
    )
    
    # setting mmr
    mmr_model = MaximalMarginalRelevance(diversity=0.3) # diversity term needed for the topic model
    
    # fitting the model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        representation_model=mmr_model,
        nr_topics=nr_topics,
        language="danish",
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=True
    )
    
    # Train the actual model
    start_time = time.time()
    topics, probs = topic_model.fit_transform(documents)
    training_time = (time.time() - start_time) / 60
    
    print(f"Complete in {training_time:.1f} min")
    print(f"   Topics: {len(set(topics))}, Outliers: {sum(t == -1 for t in topics)}")
    
    return topic_model, topics, probs


def create_topic_labels(topic_model, n_words=4):
    """Create readable topic labels from top keywords."""
    topic_labels = {}
    topic_info = topic_model.get_topic_info()
    
    for _, row in topic_info.iterrows():
        if row["Topic"] not in [-1, 0]:  # Skip outliers and the generic topic. -1 & 0 is just a shit show of thrown together common words
            words = topic_model.get_topic(row["Topic"])
            if words:
                topic_labels[row["Topic"]] = " + ".join([w[0] for w in words[:n_words]])
    
    return topic_labels