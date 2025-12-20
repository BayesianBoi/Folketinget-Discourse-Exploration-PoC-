# 🇩🇰 Folketinget Discourse Explorer

> A comprehensive tool for analyzing political discourse in the Danish Parliament (Folketinget) from 2005 to 2025 using advanced NLP and AI.

[![Logo](dashboard/logo_circle.png)](http://sorenmeiner.me/)

## 📌 Usage & purpose
**🌐 Website**: [http://sorenmeiner.me/](http://sorenmeiner.me/)

This tool is designed for **exploratory research** into Danish political rhetoric. It allows users to:
- **Track semantic trends** over time (e.g., when did "Sustainability" become a major topic?).
- **Analyze party positions** without reading thousands of speeches.
- **Find specific rhetoric** using semantic search (finding meaning, not just keywords).
- **Interact with an AI Analyst** to ask complex questions about the data.

It bridges the gap between quantitative data science (topic modeling, embeddings) and qualitative political science.

## 💾 Data Availability
The pipeline is designed to work with the **ParlLawSpeech** dataset.
- **Source**: [ParlLawSpeech Project](https://parllawspeech.org/about/)
- **Instructions**: Download the Danish corpus (`Corpus_speeches_denmark.RDS`) and place it in the `data/raw/` directory.
- **Format**: The pipeline expects the `.RDS` or `.parquet` file.

---

## 🖥️ Dashboard Features

The dashboard consists of five main analysis modules:

### 1. 📊 Overview
 **The "Big Picture" view.**
- **Discourse Over Time:** Visualize the rise and fall of specific topics (e.g., "Minkaflivning", "Skattepolitik") over the last two decades.
- **Who owns the topic?** Select a single topic to see which parties drive the debate. Does *Venstre* own the tax debate? Does *SF* own the environment debate?

### 2. 🔎 Semantic Search
 **Find speeches by meaning.**
- Unlike standard keyword search, this uses **vector embeddings** to find speeches that are *conceptually* similar to your query.
- Example: Searching for *"økonomisk ansvarlighed"* might find speeches about "budgetoverholdelse" or "statsgæld" even if the exact words aren't used.
- **Features:**
    - Filter by year range and party.
    - Download results as CSV.

### 3. 📝 Topic Inspection
 **Deep dive into the model's topics.**
- **Top Terms:** See the words that define a topic.
- **Representative Speeches:** Read the actual speeches that triggered this topic to verify the model's accuracy.
- **Word Clouds:** Visual representation of the topic's vocabulary.

### 4. 🏛️ Party Analysis
 **Profile specific parties.**
- **Topic Heatmap:** A visual matrix showing which topics different parties focus on. Who talks about what?
- **Sentiment Analysis:** (Beta) See the emotional tone of parties towards specific topics (Positive/Neutral/Negative).

### 5. 🤖 AI Analyst (BETA)
 **Ask questions in natural language.**
- Powered by OpenAI GPT-5 models, this agent has access to the processed data.
- **Capabilities:**
    - "Compare Socialdemokratiet and Venstre on tax policy in 2019."
    - "When was the peak discussion about the Mink scandal?"
    - "Give me a summary of SF's stance on climate."
- **Note:** Always verify AI claims against the charts and primary data.

---

## 🚀 Replication Guide

Follow these steps to reproduce the entire analysis from raw data to dashboard.

### 1. Prerequisites
- Python 3.10+
- A valid OpenAI API key (for the AI Agent and topic labelling)
- The raw Folketinget corpus (e.g., `Corpus_speeches_denmark.RDS` or parquet format)

### 2. Installation
```bash
# Clone the repository
git clone <repo-url>
cd Folketinget-Discourse-Exploration-PoC-

# Run the setup script (creates venv and installs dependencies)
bash scripts/setup.sh
```

### 3. Setup Configuration
1.  Place your raw data file in `data/raw/` (e.g., `data/raw/speeches.parquet`).
2.  Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=sk-your-key-here
    ```

### 4. Run the Pipeline
The entire data processing pipeline can be run with a single script:

```bash
# Activate the environment first
source .venv/bin/activate

# Run the full pipeline (steps 00-05)
bash scripts/run_pipeline.sh
```

Alternatively, you can run individual steps manually if needed:

### 5. Launch Dashboard
Once the exports are ready in `artifacts/exports/`, launch the app:

```bash
streamlit run dashboard/app.py
```

---

## 🛠️ Technical Implementation

### Core Technologies
- **Topic Modeling**: `Turftopic` with Topeax
- **Embeddings**: `intfloat/multilingual-e5-base` (State-of-the-art multilingual model)
- **Dimensionality Reduction**: UMAP
- **Backend/Processing**: Python, Pandas, Polars (for speed)
- **Frontend**: Streamlit
- **Visualization**: Plotly Interactive Charts

### Folder Structure
- `src/`: Core logic and data processing modules.
- `scripts/`: Executable pipeline steps (00-05).
- `dashboard/`: Streamlit application code.
- `artifacts/`: Generated models, plots, and data exports.
- `data/`: Raw and processed intermediate data.

---

## 👥 Authors & Credits

**Created by:** Niels Værbak & Søren Meiner

*Disclaimer: This tool is intended for exploratory research purposes. Topic models and AI interpretations are probabilistic and should be verified against primary sources.*
