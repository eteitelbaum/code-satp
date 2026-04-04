# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SATP Conflict Data Pipeline: fine-tunes transformer models (BERT variants, T5) to automatically code conflict event data scraped from the South Asia Terrorism Portal. The pipeline covers perpetrator classification, action/target type classification, location extraction with geocoding, casualty count extraction, and property damage extraction.

## Architecture

**Data flow:** SATP website → Streamlit scraper → Google Sheets → R wrangling (Quarto `.qmd`) → task-specific CSVs → model training (Colab notebooks) → fine-tuned models → inference app → coded events back to Google Sheets.

**Key components:**
- `data/` — Source dataset (`satp-dataset.xlsx`), R wrangling scripts (`.qmd`), and per-task CSVs (perpetrator.csv, action_type.csv, target_type.csv, location_info.csv, deaths.csv, etc.)
- `models/classification-models/` — Jupyter notebooks for single-label (perpetrator) and multi-label (action type, target type) classification using BERT, RoBERTa, DistilBERT, ConfliBERT, ELECTRA, XLNet. Contains `utils/` for training helpers and `imbalance-handling/` for class imbalance strategies.
- `models/location-models/` — T5 seq2seq location extraction notebooks
- `models/count-models/` — T5 QA-based count extraction (deaths, injuries, arrests, surrenders, abductions)
- `hugging-face-hosting-inference/` — Streamlit inference app (`app.py`) that orchestrates all models and geocoding. Deployed on HF Spaces. Contains fine-tuned model artifacts in subdirectories.
- `streamlit-app/` — Streamlit scraping app (`app.py`) for collecting incidents from SATP. Deployed on Streamlit Cloud.
- `papers/` and `presentations/` — Academic papers and presentations in Quarto
- `village-matching/` — Fuzzy matching for location name resolution

## Common Commands

```bash
# Python environment setup
python -m venv venv
source venv/bin/activate
pip install -r models/classification-models/requirements.txt
pip install -r streamlit-app/requirements.txt
pip install -r hugging-face-hosting-inference/requirements.txt

# Run scraping app locally
cd streamlit-app && streamlit run app.py

# Run inference app locally
cd hugging-face-hosting-inference && streamlit run app.py

# Render R/Quarto data wrangling scripts
quarto render data/wrangle_satp.qmd
quarto render data/select_vars_for_models.qmd

# Render R visualization scripts
quarto render papers/location-extraction/data-viz/bootstrap_significance_tables.R
```

## Model Training

Training runs on **Google Colab** with GPU. Notebooks clone the repo, load CSVs, and run experiments across multiple model architectures and data fractions (3%–100%). Key config pattern:

```python
EXPERIMENT_CONFIG = {
    'fractions': [1/32, 1/16, 1/8, 1/4, 1/2, 1.0],
    'models': ["bert-base-cased", "snowood1/ConfliBERT-scr-cased",
               "FacebookAI/roberta-base", "distilbert-base-cased",
               "xlnet-base-cased", "google/electra-base-discriminator"],
    'epochs': 2, 'batch_size': 32, 'max_length': 128
}
```

Production models use DistilBERT (classification) and T5 (extraction) for speed/accuracy balance.

## Key Technical Details

- **Single-label classification:** perpetrator (Security forces / Maoist / Unknown)
- **Multi-label classification:** action types, target types — uses `scikit-multilearn` utilities
- **Location extraction:** T5 seq2seq generates structured location strings, then geocoded via Google Maps API
- **Count extraction:** T5 QA approach ("How many deaths?") with regex parsing for numeric values
- **External services:** Google Sheets (gspread), Google Maps API — credentials via Streamlit secrets
- **No test suite or CI/CD** — validation is notebook-based

## Shell commands
When running Python one-liners, prefer writing a temporary script file and 
executing it rather than using python3 -c "..." with inline code. This avoids 
quote-escaping patterns that trigger security warnings.
