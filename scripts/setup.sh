#!/bin/bash
set -e

echo "[setup] Creating virtual environment (.venv)..."
python -m venv .venv
source .venv/bin/activate

echo "[setup] Upgrading pip..."
pip install --upgrade pip

echo "[setup] Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
    exit 1
fi

echo "[setup] Setup complete! Activate with: source .venv/bin/activate"
