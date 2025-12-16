#!/usr/bin/env bash
set -euo pipefail

# Run full Topeax pipeline end-to-end from repo root.
# Requires: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

echo "[run_pipeline] Step 00: validate data"
python scripts/00_validate_data.py

echo "[run_pipeline] Step 01: preprocess"
python scripts/01_preprocess.py

echo "[run_pipeline] Step 02: fit Topeax"
python scripts/02_fit_topeax.py

echo "[run_pipeline] Step 03: analyze prevalence"
python scripts/03_analyze_prevalence.py

echo "[run_pipeline] Step 04: controversial sentiment"
python scripts/04_sentiment_controversial.py

echo "[run_pipeline] Step 05: build exports"
python scripts/05_build_exports.py

echo "[run_pipeline] Step 06: visualize topics (wizard)"
python scripts/06_visualize_topicwizard.py

echo "[run_pipeline] Step 07: generate bespoke static plots"
python scripts/07_visualize_bespoke.py

echo "[run_pipeline] Done. Artifacts are under artifacts/exports/ and artifacts/plots/."
