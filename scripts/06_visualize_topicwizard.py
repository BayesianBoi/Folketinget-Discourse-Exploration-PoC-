from pathlib import Path
import sys
import joblib
import pandas as pd
import logging

try:
    import topicwizard
except ImportError:
    print("Please install topicwizard: pip install topicwizard")
    sys.exit(1)

# Add source to path for imports
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.utils.logging import setup_logging

logger = logging.getLogger(__name__)

def main():
    setup_logging()
    
    # Paths
    model_path = DEFAULT_CONFIG.paths.models_dir / "topeax_model.joblib"
    data_path = DEFAULT_CONFIG.paths.processed_dir / "preprocessed.parquet"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run 02_fit_topeax.py first.")
        return

    logger.info("Loading model...")
    model = joblib.load(model_path)
    
    logger.info("Loading data...")
    df = pd.read_parquet(data_path)
    if "text" not in df.columns:
        logger.error("Data missing 'text' column.")
        return
        
    texts = df["text"].tolist()
    
    # Topeax wrapper might be saving the inner model or the wrapper itself.
    # If the saved object is TopeaxModelRunner, we need the internal model.
    # But usually we save the model directly.
    # Let's check type.
    if hasattr(model, "model") and not hasattr(model, "transform"): # It might be TodoaxModelRunner
         logger.info("Extracting internal Turftopic model from wrapper...")
         model = model.model

    # Manual TopicData Construction to bypass broken model.transform() (due to TSNE)
    logger.info("Constructing TopicData manually...")
    
    # 1. Embed documents
    logger.info("Encoding documents...")
    embeddings = model.encode_documents(texts)
    
    # 2. Vectorize documents
    logger.info("Vectorizing documents...")
    dtm = model.vectorizer.transform(texts)
    
    # 3. Get Document-Topic Matrix
    # Since we can't run transform, we must rely on the matrix from training.
    # We can either load it from disk (doc_topics.parquet) or hope it's still attached to the model.
    if hasattr(model, "doc_topic_") and model.doc_topic_ is not None:
        doc_topic_matrix = model.doc_topic_
    else:
        # Fallback: Load from the export we made earlier
        doc_topics_path = DEFAULT_CONFIG.paths.exports_dir / DEFAULT_CONFIG.export.doc_topics_parquet
        if doc_topics_path.exists():
            logger.info(f"Loading doc-topic matrix from {doc_topics_path}...")
            dt_df = pd.read_parquet(doc_topics_path)
            # Drop doc_id col
            if "doc_id" in dt_df.columns:
                dt_df = dt_df.drop(columns=["doc_id"])
            doc_topic_matrix = dt_df.values
        else:
             logger.error("Model has no 'doc_topic_' and 'doc_topics.parquet' not found.")
             logger.error("Cannot visualize without document-topic matrix.")
             return

    # 4. Construct TopicData
    from turftopic.data import TopicData
    import numpy as np
    
    # Ensure correct shape match (in case data was reloaded/subsetted differently)
    if doc_topic_matrix.shape[0] != len(texts):
         logger.error(f"Shape mismatch: Doc-Topic matrix has {doc_topic_matrix.shape[0]} rows, but data has {len(texts)} texts.")
         return

    # Debug: Check for NaNs
    if np.isnan(embeddings).any():
        logger.warning("Embeddings contain NaNs! Filling with 0.")
        embeddings = np.nan_to_num(embeddings)
        
    if np.isnan(doc_topic_matrix).any():
        logger.warning("Doc-Topic matrix contains NaNs! Filling with 0.")
        doc_topic_matrix = np.nan_to_num(doc_topic_matrix)

    if np.isnan(model.components_).any():
        logger.warning("Topic-Term components contain NaNs! Filling with 0.")
        model.components_ = np.nan_to_num(model.components_)
    
    # DTM is sparse usually, check data if it acts like numpy
    if hasattr(dtm, "data") and np.isnan(dtm.data).any():
         logger.warning("DTM contains NaNs! Filling with 0.")
         dtm.data = np.nan_to_num(dtm.data)
    
    topic_data = TopicData(
        corpus=texts,
        vocab=model.get_vocab(),
        document_term_matrix=dtm,
        document_topic_matrix=doc_topic_matrix,
        topic_term_matrix=model.components_,
        document_representation=embeddings,
        topic_names=model.topic_names,
        transform=None, # Disable transform in UI since it's broken
    )

    logger.info("Launching topicwizard...")
    try:
        topicwizard.visualize(topic_data=topic_data)
    except Exception as e:
        logger.error(f"Failed to launch topicwizard: {e}")

if __name__ == "__main__":
    main()
