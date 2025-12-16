import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.io import load_and_validate
from dk_politics_topics.preprocess import add_time_bins, clean_dataframe
from dk_politics_topics.utils.logging import get_logger, setup_logging


def main() -> None:
    setup_logging()
    cfg = DEFAULT_CONFIG
    cfg.ensure()
    logger = get_logger(__name__)

    validated_path = cfg.paths.interim_dir / "validated.parquet"
    if validated_path.exists():
        df = pd.read_parquet(validated_path)
        logger.info("Loaded validated data from %s", validated_path)
    else:
        df, _ = load_and_validate(cfg.paths, cfg.corpus)
        logger.info("Validated data directly from raw input")

    df = clean_dataframe(df, text_col="text", cfg=cfg.preprocess)
    df = add_time_bins(df, freq="year", date_col="date", label_col="time_bin")

    output_path = cfg.paths.processed_dir / "preprocessed.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved preprocessed data to %s", output_path)


if __name__ == "__main__":
    main()
