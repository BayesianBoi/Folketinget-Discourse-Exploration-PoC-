# Danish Political Discourse: Topeax Pipeline (2005–2025)

End-to-end, reproducible pipeline for Danish Folketinget speeches using `turftopic.Topeax`, with offline batch processing now and an API-ready service layer for later (Flask/FastAPI).

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Pipeline
python scripts/00_validate_data.py
python scripts/01_preprocess.py
python scripts/02_fit_topeax.py
python scripts/03_analyze_prevalence.py
python scripts/04_sentiment_controversial.py
python scripts/05_build_exports.py
```

## Data Expectations
- Place the Folketinget corpus under `data/raw/` (default filename: `Corpus_speeches_denmark.RDS`). CSV/Parquet also work.
- Canonical columns after validation: `doc_id`, `date`, `year`, `party`, `text`, optional `source`.
- Party codes are normalized deterministically (see `CorpusConfig.party_normalization`).
- Default time window is **2005–2025**; adjust in `config.py`. Earlier years are allowed but risk:
  1. Sparse corpus size,
  2. Format drift in transcript files,
  3. Language drift (Danish spelling/terminology).

## What the pipeline produces
- `artifacts/exports/topics.json` – topic terms + labels
- `artifacts/exports/doc_topics.parquet` – doc-topic weights
- `artifacts/exports/prevalence_party_year.parquet` – per-party topic prevalence
- `artifacts/exports/controversial_sentiment_party_year.parquet` – sentiment on immigration/taxation/climate
- `artifacts/exports/metadata.json` – config + corpus stats
- `artifacts/plots/*.html` – Plotly HTML (prevalence, party comparison, controversial sentiment)
- `artifacts/exports/topic_details.csv` – topic_id, label, doc_count, top_terms (legacy-style summary)
- `artifacts/exports/party_topic_matrix.csv` – party vs. dominant topic counts (legacy-style)
- `artifacts/exports/summary_statistics.txt` – compact model/corpus summary

## Repo Structure (current)
```
src/
  dk_politics_topics/
    config.py                   # dataclasses with defaults + paths
    io/                         # loaders + schema validation
    preprocess/                 # cleaning + time bins
    modeling/                   # embeddings + Topeax wrapper
    analysis/                   # prevalence, controversial mapping, sentiment
    viz/                        # plotly builders
    service/                    # API-ready pure functions + repository access
    utils/                      # logging, caching, randomness helpers
scripts/00-05_*.py             # pipeline steps
legacy/                        # previous BERTopic pipeline (kept intact)
artifacts/                     # models, embeddings, exports, plots (gitignored)
data/{raw,interim,processed}   # datasets (gitignored)
tests/                         # pytest suite for schemas/bins/controversial/sentiment
agents.md                      # roles & hand-offs
```

## Pipeline Stages
- **00_validate_data**: load corpus (RDS/CSV/Parquet), normalize columns/party codes, basic validation, save `data/interim/validated.parquet`.
- **01_preprocess**: light cleaning (whitespace, boilerplate removal), time bins (year by default), save `data/processed/preprocessed.parquet`.
- **02_fit_topeax**: cache embeddings (SentenceTransformers), fit `turftopic.Topeax` (fallback LDA if Topeax unavailable), save model + doc-topic matrix + topics JSON. Default encoder: `all-MiniLM-L6-v2` (small, fast). If you prefer a stronger model and can tolerate longer runtimes, set `EmbeddingConfig.model_name` to `intfloat/multilingual-e5-base`; the pipeline falls back to `all-MiniLM-L6-v2` automatically.
- **03_analyze_prevalence**: per-party, per-time-bin topic prevalence.
- **04_sentiment_controversial**: match topics to immigration/taxation/climate seeds (Danish variants included), score sentiment (lexicon fallback; pluggable for HF models), aggregate by party/time.
- **05_build_exports**: assemble metadata + HTML plots for future UI/API consumption, plus interpretable CSV summaries.

## Future API Boundary
`src/dk_politics_topics/service/endpoints.py` exposes pure functions for later routes:
- `get_parties()`
- `get_topics()`
- `get_prevalence(party, topic_id, from, to, bin)`
- `compare_parties(party_a, party_b, from, to)`
- `get_controversial_sentiment(area, from, to)`

These operate on artifacts via `service/repository.py` so Flask/FastAPI can be added without refactors.

## Testing
```bash
pytest
```
Tests cover schema validation, time binning, controversial topic matching, and sentiment interface behavior (lexicon fallback; no heavy models needed).

## Legacy Code
The previous BERTopic-based approach is preserved under `legacy/`. It is not used by the current pipeline but should remain for reference/comparisons.
