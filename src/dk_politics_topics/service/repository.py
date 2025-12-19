from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..config import ExportConfig, PathsConfig
from ..utils.cache import load_json
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ArtifactRepository:
    def __init__(self, paths: PathsConfig, export_cfg: ExportConfig):
        self.paths = paths
        self.export_cfg = export_cfg

    def topics_path(self) -> Path:
        return self.paths.exports_dir / self.export_cfg.topic_json

    def doc_topics_path(self) -> Path:
        return self.paths.exports_dir / self.export_cfg.doc_topics_parquet

    def prevalence_path(self) -> Path:
        return self.paths.exports_dir / self.export_cfg.prevalence_parquet

    def controversial_path(self) -> Path:
        return self.paths.exports_dir / self.export_cfg.controversial_parquet

    def load_topics(self) -> Dict:
        path = self.topics_path()
        return load_json(path) if path.exists() else {}

    def load_prevalence(self) -> pd.DataFrame:
        path = self.prevalence_path()
        if not path.exists():
            raise FileNotFoundError(f"Missing prevalence artifact at {path}")
        return pd.read_parquet(path)

    def load_doc_topics(self) -> pd.DataFrame:
        path = self.doc_topics_path()
        if not path.exists():
            raise FileNotFoundError(f"Missing doc_topics artifact at {path}")
        return pd.read_parquet(path)

    def load_controversial_sentiment(self) -> pd.DataFrame:
        path = self.controversial_path()
        if not path.exists():
            raise FileNotFoundError(f"Missing controversial sentiment artifact at {path}")
        return pd.read_parquet(path)

    def list_parties(self) -> List[str]:
        prev = self.load_prevalence()
        return sorted(prev["party"].dropna().unique().tolist())
