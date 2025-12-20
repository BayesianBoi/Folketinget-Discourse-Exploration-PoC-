#!/bin/bash

# Create the virtual environment
python3 -m venv env

# Activating it
source env/bin/activate

# Upgrade pip (I just copied this from a previous setup script I had)
pip install --upgrade pip

# Install the requirements
echo "Installing the requirements"
pip install -r "requirements.txt"

# Download the danish model
echo "Downloading spacy Danish model..."
python -m spacy download da_core_news_sm

echo "Setup complete!"
echo ""
echo "To activate the environment, just run:"
echo "source env/bin/activate"