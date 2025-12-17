from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .utils.stopwords import get_combined_stopwords

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def project_root() -> Path:
    """Return repository root relative to this file."""
    return Path(__file__).resolve().parents[2]


@dataclass
class PathsConfig:
    base_dir: Path = field(default_factory=project_root)
    data_dir: Optional[Path] = None
    raw_dir: Optional[Path] = None
    interim_dir: Optional[Path] = None
    processed_dir: Optional[Path] = None
    artifacts_dir: Optional[Path] = None
    embeddings_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    exports_dir: Optional[Path] = None
    plots_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        self.data_dir = self.data_dir or (self.base_dir / "data")
        self.raw_dir = self.raw_dir or (self.data_dir / "raw")
        self.interim_dir = self.interim_dir or (self.data_dir / "interim")
        self.processed_dir = self.processed_dir or (self.data_dir / "processed")
        self.artifacts_dir = self.artifacts_dir or (self.base_dir / "artifacts")
        self.embeddings_dir = self.embeddings_dir or (self.artifacts_dir / "embeddings")
        self.models_dir = self.models_dir or (self.artifacts_dir / "models")
        self.exports_dir = self.exports_dir or (self.artifacts_dir / "exports")
        self.plots_dir = self.plots_dir or (self.artifacts_dir / "plots")

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.embeddings_dir,
            self.models_dir,
            self.exports_dir,
            self.plots_dir,
        ]:
            if path:
                path.mkdir(parents=True, exist_ok=True)


@dataclass
class CorpusConfig:
    start_year: int = 2005
    end_year: int = 2025
    allow_earlier: bool = True
    min_text_length: int = 60
    sample_size: Optional[int] = 20000
    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "speech_id": "doc_id",
            "speechtext": "text",
            "party_name": "party",
        }
    )
    party_normalization: Dict[str, str] = field(
        default_factory=lambda: {
            "SOCIALDEMOKRATIET": "S",
            "VENSTRE": "V",
            "DANMARKSDEMOKRATERNE": "DD",
            "LIBERAL ALLIANCE": "LA",
            "ENHEDSLISTEN": "EL",
            "DET KONSERVATIVE FOLKEPARTI": "KF",
            "SOCIALISTISK FOLKEPARTI": "SF",
            "DANSK FOLKEPARTI": "DF",
            "NYE BORGELIGE": "NB",
            "RADIKALE VENSTRE": "RV",
            "INUIT ATAQATIGIIT": "IA",
            "KRISTENDEMOKRATERNE": "KD",
            "MODERATERNE": "M",
            "NY ALLIANCE": "NY",
            "FRIE GRØNNE": "FG",
            "JAVNAÐARFLOKKURIN": "JF",
            "NUNATTA QITORNAI": "NQ",
            "SIUMUT": "SIU",
            "SAMBANDSFLOKKURIN": "SP",
            "YEAH": "YEAH", # Just in case
        }
    )
    doc_id_prefix: str = "dk"

@dataclass
class PreprocessConfig:
    lowercase: bool = False
    remove_boilerplate_phrases: List[str] = field(
        default_factory=lambda: [
            "hr. formand",
            "fru formand",
            "mødet er hævet",
        ]
    )
    keep_characters: str = "æøåÆØÅ"
    min_words: int = 5
    ignored_speakers: List[str] = field(
        default_factory=lambda: [
            "Formanden",
            "Første næstformand",
            "Anden næstformand",
            "Tredje næstformand",
            "Fjerde næstformand",
            "Aldersformanden",
            "Midlertidig formand",
        ]
    )
    ignored_parties: List[str] = field(default_factory=lambda: ["-"])


@dataclass
class EmbeddingConfig:
    # Default to a public, high-quality multilingual model. If you have access to
    # a stronger Danish encoder (e.g., chcaa/dfm-encoder-large-v1 from the
    # Scandinavian embedding benchmark), set model_name to that.
    # Benchmark-informed choice: multilingual-e5-base offers strong Scandinavian performance
    # without the heavy footprint of larger TTC-L2V variants.
    model_name: str = "intfloat/multilingual-e5-small"
    model_fallback: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64  # smaller model supports larger batch on MPS
    device: str = "auto"  # auto -> mps if available, else cuda, else cpu
    cache_embeddings: bool = True
    cache_name: Optional[str] = None
    show_progress: bool = True


@dataclass
class TopeaxConfig:
    perplexity: int = 50
    random_state: int = 42
    min_df: int = 2
    max_features: int = 8000
    top_k_terms: int = 12
    embedding_model: Optional[str] = "intfloat/multilingual-e5-small"
    use_fallback_if_missing: bool = True
    verbose: bool = True
    stopwords: List[str] = field(default_factory=get_combined_stopwords)
    openai_model: Optional[str] = "gpt-5-mini"
    ignored_topics: List[int] = field(default_factory=list)  # Topics to hide in dashboard


@dataclass
class SentimentConfig:
    approach: str = "huggingface"
    huggingface_model: Optional[str] = "alexandrainst/da-sentiment-base"
    batch_size: int = 32
    device: str = "auto"  # auto -> mps/cuda if available


@dataclass
class ExportConfig:
    topic_json: str = "topics.json"
    doc_topics_parquet: str = "doc_topics.parquet"
    prevalence_parquet: str = "prevalence_party_year.parquet"
    controversial_parquet: str = "controversial_sentiment_party_year.parquet"
    metadata_json: str = "metadata.json"


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    topeax: TopeaxConfig = field(default_factory=TopeaxConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def as_dict(self) -> Dict:
        return asdict(self)

    def ensure(self) -> None:
        self.paths.ensure_dirs()


DEFAULT_CONFIG = PipelineConfig()
