import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.analysis import compute_prevalence
from dk_politics_topics.utils.logging import get_logger, setup_logging


def main() -> None:
    setup_logging()
    cfg = DEFAULT_CONFIG
    cfg.ensure()
    logger = get_logger(__name__)

    doc_topics_path = cfg.paths.exports_dir / cfg.export.doc_topics_parquet
    data_path = cfg.paths.processed_dir / "preprocessed.parquet"
    if not doc_topics_path.exists():
        raise FileNotFoundError("Run scripts/02_fit_topeax.py before analyzing prevalence.")
    if not data_path.exists():
        raise FileNotFoundError("Run scripts/01_preprocess.py before analyzing prevalence.")

    doc_topics = pd.read_parquet(doc_topics_path)
    metadata = pd.read_parquet(data_path)[["doc_id", "party", "time_bin", "year"]]

    prevalence = compute_prevalence(doc_topics, metadata, time_bin_col="time_bin")
    prevalence_path = cfg.paths.exports_dir / cfg.export.prevalence_parquet
    prevalence.to_parquet(prevalence_path, index=False)
    logger.info("Saved prevalence to %s", prevalence_path)


if __name__ == "__main__":
    main()
