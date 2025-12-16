import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.io import load_and_validate
from dk_politics_topics.utils.logging import get_logger, setup_logging


def main() -> None:
    setup_logging()
    cfg = DEFAULT_CONFIG
    cfg.ensure()
    df, issues = load_and_validate(cfg.paths, cfg.corpus)
    interim_path = cfg.paths.interim_dir / "validated.parquet"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_path, index=False)
    report_path = cfg.paths.interim_dir / "validation_report.json"
    report_path.write_text(json.dumps({"issues": issues, "rows": len(df)}, indent=2, ensure_ascii=False), encoding="utf-8")
    logger = get_logger(__name__)
    logger.info("Saved validated data to %s", interim_path)
    logger.info("Validation issues: %s", issues if issues else "none")


if __name__ == "__main__":
    main()
