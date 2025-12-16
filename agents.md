# Agents and Responsibilities

## Data Engineer
- **Owns:** `scripts/00_validate_data.py`, `scripts/01_preprocess.py`, `src/dk_politics_topics/io/*`, `src/dk_politics_topics/preprocess/*`, `src/dk_politics_topics/utils/*`
- **Input:** Raw corpus in `data/raw/`, configuration in `config.py`
- **Output:** Validated/interim data (`data/interim/validated.parquet`), preprocessed corpus (`data/processed/preprocessed.parquet`)
- **Notes:** Keeps path handling relative, maintains encoding safety for Danish characters, ensures caching directories exist.

## NLP / Topic Modeling
- **Owns:** `src/dk_politics_topics/modeling/*`, `scripts/02_fit_topeax.py`
- **Input:** Preprocessed corpus, embeddings
- **Output:** Topeax model artifact (`artifacts/models/topeax_model.joblib`), topic terms JSON, doc-topic matrices
- **Notes:** Default to deterministic seeds, prefer cached embeddings, avoid deleting anything in `legacy/`.

## Analysis / Controversial & Sentiment
- **Owns:** `src/dk_politics_topics/analysis/*`, `scripts/03_analyze_prevalence.py`, `scripts/04_sentiment_controversial.py`
- **Input:** Doc-topic matrix, topic terms, preprocessed metadata
- **Output:** Prevalence tables, controversial topic mappings, sentiment aggregates
- **Notes:** Keep controversial seeds transparent; sentiment provider must be pluggable (lexicon fallback if transformers unavailable).

## Visualization
- **Owns:** `src/dk_politics_topics/viz/*`, plot outputs in `artifacts/plots/`, `scripts/05_build_exports.py`
- **Input:** Prevalence and sentiment aggregates
- **Output:** Plotly HTML plots for the future UI/API to serve
- **Notes:** Avoid hardcoded colors; ensure plots write to disk and are API-friendly (HTML + JSON-ready).

## Backend / API
- **Owns:** `src/dk_politics_topics/service/*`
- **Input:** Saved artifacts (`artifacts/exports/*`)
- **Output:** Pure functions usable by future Flask/FastAPI endpoints (`get_topics`, `get_prevalence`, etc.)
- **Notes:** Keep interfaces stable, avoid side effects, validate inputs lightly.

## QA / Testing
- **Owns:** `tests/`
- **Input:** Unit tests for schema validation, time binning, controversial matching, sentiment provider
- **Output:** Fast-running pytest suite; regression safety for pipeline contracts
- **Notes:** Tests should run without heavy models (mock or fallback paths).

## Workflow
1. **Data Engineer** runs validation (`00_validate_data.py`) and preprocessing (`01_preprocess.py`), delivers cleaned Parquet.
2. **NLP/Topic Modeling** fits Topeax (`02_fit_topeax.py`), saves topics + doc-topic matrix.
3. **Analysis** computes prevalence (`03_analyze_prevalence.py`) and controversial sentiment (`04_sentiment_controversial.py`).
4. **Visualization** builds exports and plots (`05_build_exports.py`).
5. **Backend/API** consumes artifacts via `service/repository.py` and exposes pure functions for future routes.
6. **QA** runs `pytest` to ensure schemas, binning, controversial matching, and sentiment interfaces behave.

## Coding Conventions
- Typing and dataclasses for configs; keep random seeds in `config.py`.
- Logging via `dk_politics_topics.utils.logging`; no print statements in pipeline code.
- Paths resolved relative to repo root via `PathsConfig`; no absolute paths.
- Cache heavy artifacts under `artifacts/` and keep raw data out of git.
- Preserve Danish characters; cleaning should be minimal and documented.
