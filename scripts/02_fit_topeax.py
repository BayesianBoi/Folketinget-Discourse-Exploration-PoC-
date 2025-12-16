import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.modeling import TopeaxModelRunner, embed_texts, label_topics
from dk_politics_topics.utils.cache import save_json
from dk_politics_topics.utils.logging import get_logger, setup_logging


def main() -> None:
    setup_logging()
    cfg = DEFAULT_CONFIG
    cfg.ensure()
    logger = get_logger(__name__)

    data_path = cfg.paths.processed_dir / "preprocessed.parquet"
    if not data_path.exists():
        raise FileNotFoundError("Run scripts/01_preprocess.py before fitting the model.")
    df = pd.read_parquet(data_path)

    logger.info("Preparing embeddings for %d documents", len(df))
    embeddings = embed_texts(df["text"], cfg.paths, cfg.embedding)
    logger.info("Embeddings ready; starting Topeax fit")
    runner = TopeaxModelRunner(cfg.topeax, cfg.paths)
    doc_topics, topic_terms = runner.fit(df["text"], df["doc_id"], embeddings)

    doc_topics_path = cfg.paths.exports_dir / cfg.export.doc_topics_parquet
    doc_topics_path.parent.mkdir(parents=True, exist_ok=True)
    doc_topics.to_parquet(doc_topics_path, index=False)

    labels = {}
    # Use LLM topic names if available from the internal Turftopic model
    if hasattr(runner.model, "topic_names") and runner.model.topic_names:
        # topic_names is usually a list where index corresponds to topic ID
        for idx, name in enumerate(runner.model.topic_names):
            labels[str(idx)] = name
    else:
        # Fallback to top-3 terms
        labels = label_topics(topic_terms)
    topics_payload = {
        "topics": topic_terms,
        "metadata": {"labels": labels, "config": cfg.as_dict()},
    }
    topics_path = cfg.paths.exports_dir / cfg.export.topic_json
    save_json(topics_payload, topics_path)

    runner.save_model()

    logger.info("Saved doc-topic matrix to %s", doc_topics_path)
    logger.info("Saved topics to %s", topics_path)


if __name__ == "__main__":
    main()
