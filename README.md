# Financial Sentiment Analysis Pipeline

Production-ready sentiment analysis for financial news with **81.8% accuracy**. Automated pipeline for fetching, analyzing, and reporting sentiment from financial news articles.

##  Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 81.8% |
| **F1 Macro** | 79.3% |
| **F1 Negative** | 65.8% |
| **F1 Neutral** | 84.6% |
| **F1 Positive** | 87.4% |

## Features

- ✅ **Fine-Tuned FinBERT** - Trained on FinancialPhraseBank (5,842 expert-labeled sentences)
- ✅ **NewsAPI Integration** - Real-time financial news fetching
- ✅ **Web UI Interface** - User-friendly Gradio interface (no command line needed!)
- ✅ **Class-Weighted Training** - Handles imbalanced data effectively
- ✅ **Daily Aggregation** - Sentiment metrics with rolling trends
- ✅ **Alert System** - Automated spike detection (z-score > 2.5)
- ✅ **Dual Format Support** - Both Parquet and CSV output

## 🎨 Quick Start with Web UI

**Easiest way to use the pipeline!**

```powershell
# Install dependencies
pip install -r requirements.txt

# Set API key
$env:NEWSAPI_KEY="your_newsapi_key_here"

# Launch web interface
python app.py
```

Open browser at **http://127.0.0.1:7860** and use the web interface to:
1. **Fetch News** - Enter company/ticker and download articles
2. **Analyze Sentiment** - Generate reports with one click
3. **Test Sentences** - Try the model on custom text

See **[UI_GUIDE.md](UI_GUIDE.md)** for complete UI documentation.

## 💻 Command Line Usage

## 💻 Command Line Usage

### Prerequisites
- Python 3.12+
- NewsAPI key (get free at https://newsapi.org/)

### Option 1: Using uv (Recommended)

```powershell
# Install dependencies
uv sync

# Set API key
$env:NEWSAPI_KEY="your_newsapi_key_here"
```

### Option 2: Using venv

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set API key
$env:NEWSAPI_KEY="your_newsapi_key_here"
```

## Usage

### 1. Fetch Financial News

```powershell
# Fetch news for Apple (AAPL)
uv run python -m src.cli.main `
    --company "Apple Inc" `
    --ticker AAPL `
    --keywords "iPhone,services" `
    --days 7 `
    --max 150 `
    --format csv `
    --verbose

# Output: data/processed/YYYYMMDD/aapl_YYYYMMDD.csv
```

### 2. Analyze Sentiment

```powershell
# Generate sentiment reports
uv run python -m src.cli.aggregate `
    --dataset data/processed/20251007/aapl_20251007.csv `
    --ticker AAPL `
    --out reports `
    --window 7 `
    --format csv

# Output:
# - reports/aapl/daily.csv (daily sentiment metrics)
# - reports/aapl/alerts.csv (sentiment spikes)
```

### 3. Run Tests

```powershell
uv run pytest -q
```

## Pipeline Phases

1. **Ingestion** - Fetch articles from NewsAPI, clean text, deduplicate
2. **Model** - Pre-trained FinBERT model included (models/finetuned_improved/)
3. **Aggregation** - Daily sentiment metrics with rolling trends and anomaly detection

## Optional: Train Your Own Model

```powershell
# Download FinancialPhraseBank dataset
python download_financialphrasebank.py --format csv

# Train on Google Colab (recommended - 3-5 minutes)
# See COMPLETE_DOCUMENTATION.md Section 7 for Colab guide

# Or train locally (20-30 minutes)
uv run python -m src.cli.train_model `
    --dataset data/datasets/financialphrasebank.csv `
    --model_name ProsusAI/finbert `
    --out models/finetuned `
    --epochs 2 `
    --batch 8 `
    --label-col label
```

## File Formats

- **Parquet** (default): Faster, smaller - recommended for production
- **CSV**: Excel-compatible - good for inspection

Use --format csv flag or .csv extension in output path.

## Documentation

See **[COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)** for:
- Full architecture and workflow
- Model training journey (5 optimization phases)
- Google Colab training guide
- Class weights for imbalanced data
- Model comparison and selection
- Troubleshooting guide

## Project Structure

```
Sentiment/
 src/                          # Source code
    cli/                      # Command-line interfaces
    data/loaders/             # NewsAPI integration
    models/                   # ML training & inference
    aggregation/              # Report generation
 models/finetuned_improved/    # Production model 
 tests/                        # Unit tests
 COMPLETE_DOCUMENTATION.md     # Full documentation
```

## Model Details

- **Base**: ProsusAI/finbert
- **Training**: FinancialPhraseBank (5,842 expert-labeled sentences)
- **Platform**: Google Colab (T4 GPU, 3-5 minutes)
- **Techniques**: Class-weighted loss, regularization, early stopping
- **Performance**: 81.8% accuracy, 79.3% F1 macro

## Troubleshooting

- **Slow training?**  Use Google Colab (50x faster)
- **CSV errors?**  Fixed with ast.literal_eval() for ticker parsing
- **Poor minority class?**  Model uses class weights (implemented)

Full troubleshooting in [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md) Section 11.

---

**Status**:  Production-ready  
**Last Updated**: October 7, 2025  
**Model**: models/finetuned_improved/ (81.8% accuracy)
