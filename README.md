# Movie Review Translation & Sentiment Analysis

A mini-pipeline that takes movie reviews written in English, French and Spanish, automatically translates everything to English with Transformer-based machine-translation models and finally performs sentiment analysis to decide whether each review is **Positive** or **Negative**.

The whole workflow is implemented in a single Jupyter notebook: **`movie_sentiment_analysis.ipynb`**.

---

## 1. Project goals

1. **Data ingestion** – Load three CSV files (one per language) containing the movie *Title*, *Year*, *Synopsis* and *Review*.
2. **Pre-processing** – Standardise column names and keep track of the original language.
3. **Machine Translation** –
   * French → English model: `Helsinki-NLP/opus-mt-fr-en`
   * Spanish → English model: `Helsinki-NLP/opus-mt-es-en`
4. **Sentiment Analysis** – `distilbert-base-uncased-finetuned-sst-2-english` fine-tuned BERT classifier from 🤗 Transformers.
5. **Export** – Save one tidy data set (`result/reviews_with_sentiment.csv`) containing 30 movies with their translated reviews and predicted sentiment.

> The notebook is fully reproducible; just **Run-All** and wait for the models to download the first time.

---

## 2. Repository layout

```
├── data/               # Raw CSV files (EN/FR/ES)
├── images/             # Optional figures used in markdown notes
├── lectures/           # Independent demo notebooks (BERT, LSTM, etc.)
├── result/             # Notebook output – translated reviews + sentiment
├── movie_sentiment_analysis.ipynb  # 🚀 Main notebook
├── requirements.txt    # Python dependencies (pip)
├── environment.yaml    # Conda environment (optional)
└── README.md           # You are here ✨
```

### The `lectures/` folder
Additional notebooks used during the Nanodegree for experimenting with tokenisation, LSTM classifiers, BERT embeddings, time-series preprocessing, etc.  They are **not** required to reproduce the main project but may serve as learning material.

---

## 3. Quick start

### 3.1 Clone & install

```bash
# clone the repo
git clone <repo-url>  # or download the zip
cd 3_movie_sentiment_analysis

# OPTION A – Conda (recommended)
conda env create -f environment.yaml
conda activate rnn

# OPTION B – Pip
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 Download NLTK data (optional)
Some auxiliary notebooks import `nltk`.  If you run them you may need to download corpora:

```python
import nltk; nltk.download('punkt')
```

### 3.3 Run the notebook

1. Launch Jupyter Lab/Notebook:
   ```bash
   jupyter lab   # or: jupyter notebook
   ```
2. Open **`movie_sentiment_analysis.ipynb`**.
3. Run all cells (⏵ ▶ or Kernel → **Restart & Run All**).

The first execution will download ~1 GB of pretrained models from Hugging Face and may take a few minutes depending on your connection.

Once finished, the translated and annotated data set is written to:

```
result/reviews_with_sentiment.csv
```

A preview of the resulting file:

| Title | Year | Review (EN) | Sentiment |
|-------|------|-------------|-----------|
| The Shawshank Redemption | 1994 | The Shawshank Redemption is an inspiring tale … | Positive |
| Blade Runner 2049 | 2017 | Boring and too long. Nothing like the original … | Negative |
| … | … | … | … |

---

## 4. Key implementation details

* **Vectorised translation:** Instead of translating field-by-field with explicit for-loops, the notebook uses `.loc[mask, col].apply(...)` to keep the code readable while still benefitting from pandas' internal optimisation.
* **Device detection:**  Automatically selects *MPS* (Apple Silicon), *CUDA* or *CPU* depending on what is available.
* **Text cleaning:**  Simple utility removes leading/trailing quotes and drops empty reviews before classification.
* **Deterministic column order & row count:**  The final CSV is trimmed to **30 rows** and validated before saving.

---

## 5. Extending the project

* Swap in different translation or sentiment models by just changing the model names.
* Add more languages by extending the `translation_configs` dictionary.
* Use batch translation for faster throughput on large data sets (`model.generate` supports batched input).
* Fine-tune the sentiment classifier on movie-specific data to improve accuracy.

---

## 6. Acknowledgements

* [Helsinki-NLP OPUS-MT](https://huggingface.co/Helsinki-NLP) – open-source machine-translation models.
* [🤗 Transformers](https://github.com/huggingface/transformers) – unified API to state-of-the-art NLP models.
* Udacity *Deep Learning Nanodegree* – project specification.

---

## 7. License

This repository is released for educational purposes.  See `LICENSE` (to be added) for full details.
