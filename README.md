# Aspect-Based Sentiment Analysis (Two-Stage) Inference Project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12AqOSj-AEPnaTIht3LYqSLhUhws2dOt4?usp=sharing)

This repository is a production-style refactor of the original Colab notebook above, with GitHub Copilot support for code organization, refactoring, and building inference apps (FastAPI + Streamlit).

This is a **two-stage ABSA pipeline**:
- **Stage 1 (ABTE):** extract aspect terms from the sentence.
- **Stage 2 (ABSC):** predict the sentiment for each extracted term.

You can download the pre-trained model weights from this [Google Drive folder](https://drive.google.com/drive/folders/1mGfYKVOJNzUYQPw5QC7ZeExiWHTikxmN?usp=sharing) and place them in the `saved_models/` directory.

## 📁 Project Structure

```bash
.
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   ├── __init__.py
│   │   └── main.py               # API endpoints
│   ├── models/                   # Model loading & inference
│   │   ├── __init__.py
│   │   └── model_loader.py       # Model classes and loader
│   ├── config/                   # Configuration
│   │   ├── __init__.py
│   │   └── config.py             # App settings
│   └── __init__.py
├── ui/                           # Streamlit frontend
│   └── app.py                    # UI application
├── notebooks/                    # Jupyter notebooks and exported scripts
├── saved_models/                 # Pre-trained model weights
│   ├── abte-bilstm/
│   ├── abte-minilm/
│   ├── absc-bilstm/
│   └── absc-minilm/
├── requirements.txt              # Python dependencies
├── Makefile                      # Commands for easy execution
└── README.md                     # This file
```

## ⚙️ Installation

### 1. Install Dependencies

```bash
make install
```

Or manually:

```bash
pip install -r requirements.txt
```

### 2. Verify Model Files

Ensure all model weights exist in `saved_models/`:

```bash
ls -la saved_models/
```

## 🏃 Running the Application

### Option 1: Run Both Servers (Recommended)

```bash
make run
```

This starts:
- API Server: http://127.0.0.1:8000
- Streamlit UI: http://localhost:8501

### Option 2: Run Separately

```bash
# Terminal 1: Start FastAPI
make run-api

# Terminal 2: Start Streamlit
make run-ui
```

### Option 3: Manual Run

```bash
# API Server
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Streamlit UI
streamlit run ui/app.py --server.port=8501 --server.address=localhost
```

## 📊 ABTE Labels

- O (0): Non-aspect token
- B-Term (1): Begin of aspect term
- I-Term (2): Inside of aspect term

## 📝 Model Specifications

### Custom LSTM (ABTE + ABSC)

- Configuration is loaded from each model folder under `saved_models/` (see `config.json`).
- The service reads those configs to build the correct LSTM architecture for ABTE/ABSC.
- Tokenization: word-level tokenizer.
- Max sequence length: configurable via `LSTM_MAX_LENGTH` (default: 128).

### MiniLM (ABTE + ABSC)

- Use `sentence-transformers/all-MiniLM-L6-v2` or other MiniLM variants such as L12.
- Typical parameters: hidden size, number of layers, and max length come from the selected checkpoint.

## 📊 Benchmark Performance

F1 is the primary metric. Scores below are on the test split.

### ABSC (Sentiment Classification)

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| BiLSTM ABSC | 0.6577 | 0.5816 | 0.5871 | 0.5720 |
| MiniLM ABSC | 0.8490 | 0.7966 | 0.7585 | 0.7735 |

### ABTE (Term Extraction)

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| BiLSTM ABTE | 0.9271 | 0.7524 | 0.7784 | 0.7652 |
| MiniLM ABTE | 0.9595 | 0.8629 | 0.8714 | 0.8671 |

## 🌐 API Endpoints

- GET `/health`: health check
- GET `/models`: list ABTE/ABSC models under `saved_models/`
- POST `/predict`: run ABSA inference (ABTE -> ABSC)

Request example:

```json
{
  "text": "The bread is top notch as well",
  "abte_model_name": "abte-minilm",
  "absc_model_name": "absc-minilm",
  "term": null,
  "device": "auto"
}
```
