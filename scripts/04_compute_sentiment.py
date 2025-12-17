import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.analysis.sentiment import resolve_provider, score_documents
from dk_politics_topics.utils.logging import get_logger, setup_logging

def main() -> None:
    setup_logging()
    cfg = DEFAULT_CONFIG
    cfg.ensure()
    logger = get_logger(__name__)

    # 1. Check Previous Steps
    data_path = cfg.paths.processed_dir / "preprocessed.parquet"
    if not data_path.exists():
        raise FileNotFoundError("Ensure Step 01 (preprocessing) has been run.")

    # 2. Load Data (Metadata/Texts)
    logger.info("Loading preprocessed data from %s...", data_path)
    df = pd.read_parquet(data_path)
    
    # 3. Setup Provider
    provider = resolve_provider(cfg.sentiment)
    logger.info("Initialized sentiment provider: %s", type(provider).__name__)
    
    # 4. Score Documents
    logger.info("Scoring %d documents...", len(df))
    # Using 'text' column from preprocessed data
    df_scored = score_documents(df, "text", provider)
    
    # 5. Save Scores
    output_path = cfg.paths.exports_dir / "sentiment_scores.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save minimal columns to save space/time
    df_scored[["doc_id", "sentiment"]].to_parquet(output_path, index=False)
    logger.info("Saved sentiment scores to %s", output_path)

if __name__ == "__main__":
    main()
