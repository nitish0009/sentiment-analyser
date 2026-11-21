# Sentiment Analysis System

A complete, reproducible **binary sentiment analysis** project built with classical
NLP and machine learning. It trains on the real **IMDb 50K movie-review dataset**,
compares **Multinomial Naive Bayes** and **Logistic Regression**, and serves the
selected model through a polished **Streamlit** application.

The model predicts **Positive** or **Negative** and reports an honest confidence
score. It does **not** invent a Neutral class: the dataset has no neutral labels,
so a neutral prediction would not be supported by the training data.

---

## Highlights

- **Real data, real metrics** — 50,000 IMDb reviews, no hard-coded scores.
- **Leakage-free feature extraction** — TF-IDF is fitted on the training split only.
- **Shared preprocessing** — training and prediction call the exact same function.
- **Two models compared** — Multinomial Naive Bayes and Logistic Regression.
- **Full evaluation** — accuracy, precision, recall, F1, per-class reports and
  confusion matrices.
- **Saved artifacts** — the app loads `model.pkl` and `vectorizer.pkl`; it never
  retrains while a user is waiting.
- **Polished 4-page UI** — Home, Analyzer, Model Performance, Sample Reviews.
- **Low-confidence warning** — below 60% confidence the interface says so instead
  of pretending the prediction is certain.
- **17 automated tests** — preprocessing and saved-artifact checks.

---

## Quick start

If starting from a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt

python scripts/setup_nltk.py
python scripts/download_dataset.py
python src/train.py

python -m streamlit run streamlit_app.py
```

If the `models/` directory already contains the trained artifacts, you can skip
the download and training steps and run the app directly. The full setup notes,
manual dataset instructions and troubleshooting are below.

---

## Measured results

Stratified 80/20 split, `random_state=42`. The test set contains **9,917** reviews
(4,940 negative, 4,977 positive).

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 87.79% | 87.80% | 87.79% | 87.79% |
| **Logistic Regression** (selected) | **89.65%** | **89.68%** | **89.65%** | **89.65%** |

Selected model: **Logistic Regression**, chosen by the highest weighted F1 on the
held-out test set (accuracy is the tie-breaker).

Confusion matrix for Logistic Regression (rows = true, columns = predicted):

|  | Predicted Negative | Predicted Positive |
|---|---:|---:|
| **True Negative** | 4,368 | 572 |
| **True Positive** | 454 | 4,523 |

The complete evaluation is in [`evaluation_report.md`](evaluation_report.md).

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.14.4 (3.12+ recommended) |
| Data | pandas, NumPy |
| NLP | NLTK (`word_tokenize`, stop-words, Porter stemmer) |
| Features | scikit-learn `TfidfVectorizer` (unigrams + bigrams) |
| Models | `MultinomialNB`, `LogisticRegression` |
| Evaluation | scikit-learn metrics, Matplotlib, Seaborn |
| Persistence | Joblib |
| UI | Streamlit 1.60.0 |
| Tests | `unittest` |

No BERT, transformers, LLMs, paid APIs or external inference services are used.

---

## Project structure

```text
sentiment analyzer/
├── streamlit_app.py          # Streamlit UI for local demonstrations
├── server.py                 # Flask API used by the Vercel deployment
├── vercel.json               # Vercel function bundle configuration
├── .python-version           # Python 3.14 for Vercel
├── requirements.txt          # Slim runtime deps used by Vercel
├── requirements-local.txt    # Streamlit + training dependencies
├── README.md
├── evaluation_report.md      # Detailed measured evaluation
├── .streamlit/config.toml    # Theme + usage-statistics setting
├── public/
│   ├── index.html            # Vercel-compatible static analyzer UI
│   ├── styles.css
│   ├── app.js
│   └── reports/              # Evaluation charts served by the CDN
├── nltk_data/                # English stop-words + punkt_tab bundle
├── data/
│   ├── README.md              # Dataset download notes
│   └── dataset.csv            # Generated locally; ignored by git
├── docs/
│   ├── PROJECT_REPORT.md      # Full project report (12 sections)
│   ├── VIVA_QA.md             # 36 viva questions and answers
│   ├── DEMO_SCRIPT.md         # 5–7 minute demonstration plan
│   └── SCREENSHOTS.md         # Optional screenshot capture guide
├── models/
│   ├── sentiment_model.pkl    # Saved classifier
│   ├── tfidf_vectorizer.pkl   # Saved TF-IDF vocabulary + weights
│   └── model_metadata.json    # Real metrics, split, settings, versions
├── reports/
│   ├── class_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrix_*.png
│   ├── top_features.png
│   ├── classification_report_*.txt
│   ├── dataset_sample.txt
│   ├── preprocessing_examples.txt
│   └── training_run.log
├── scripts/
│   ├── download_dataset.py    # Download IMDb, extract, build dataset.csv
│   └── setup_nltk.py          # Download required NLTK resources
├── src/
│   ├── config.py              # Paths and shared constants
│   ├── data_utils.py          # CSV validation and duplicate removal
│   ├── preprocessing.py       # Shared cleaning pipeline
│   ├── nltk_setup.py          # NLTK resource checks
│   ├── evaluation.py          # Metrics and charts
│   ├── train.py               # End-to-end training pipeline
│   └── predict.py             # CLI prediction wrapper
└── tests/
    ├── test_preprocessing.py
    └── test_artifacts.py
```

---

## Setup

### 1. Prerequisites

- Python **3.12+** (tested with 3.14.4)
- About **1.5 GB** free disk space for the raw dataset, CSV, models and charts
- macOS, Linux or Windows

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-local.txt
```

> If `streamlit` is not on your PATH, use `python -m streamlit` instead of
> `streamlit` in every command below. The project was tested both ways.

### 3. Download the NLTK resources

```bash
python scripts/setup_nltk.py
```

This downloads `punkt_tab` and `stopwords`. `wordnet` is not needed. NLTK 3.10
requires the data directory to be group/world non-writable:

```bash
chmod -R go-w ~/nltk_data
```

### 4. Build the dataset

```bash
python scripts/download_dataset.py
```

The script downloads `aclImdb_v1.tar.gz` from Stanford (about 84 MB), extracts
the labelled train/test folders, and writes a shuffled `data/dataset.csv` with
two columns: `review` and `sentiment`. Use `--keep-raw` if you want to keep the
extracted files for inspection. If `data/dataset.csv` already exists, the script
stops early; use `--force` to rebuild it.

If you already have `data/raw/aclImdb`, the script detects it and skips the
download.

### 5. Train and evaluate

```bash
python src/train.py
```

This takes roughly **1–2 minutes** on a laptop. It:

1. removes missing, empty, invalid and duplicate rows;
2. creates the stratified 80/20 split;
3. preprocesses both splits with the shared pipeline;
4. fits TF-IDF on the **training split only**;
5. trains both models, calculates metrics, and saves charts;
6. selects the best weighted F1 and saves the artifacts.

You can change the test fraction or feature count:

```bash
python src/train.py --test-size 0.2 --max-features 50000
```

### 6. Run the application

```bash
python -m streamlit run streamlit_app.py
```

Then open <http://localhost:8501>.

If the browser does not open automatically, copy the URL printed in the
terminal. Stop the server with `Ctrl+C`.

---

## Using the application

### Home
An overview of the project, the real model snapshot, a pipeline diagram and a
short usage guide.

### Sentiment Analyzer
1. Paste a review (up to 5,000 characters) or load an example.
2. Press **Analyze sentiment**.
3. Read the predicted class, confidence, and the negative/positive probability bars.
4. Expand **See preprocessing details** to show the exact cleaned token string
   that was passed to TF-IDF — useful in a viva.
5. If confidence is below **60%**, a clear low-confidence warning appears.

### Model Performance
- Selected model banner and headline metrics.
- Naive Bayes vs Logistic Regression chart and table.
- Side-by-side confusion matrices with TN/FP/FN/TP captions.
- Per-class classification reports.
- The strongest learned positive and negative tokens.
- Dataset, split and reproducibility details.

### Sample Reviews
Six clear positive/negative examples plus three tricky cases. Every card loads
the text into the analyzer and runs the saved model automatically.

---

## Deploy on Vercel

Vercel cannot host the Streamlit server itself because Streamlit needs a
long-lived WebSocket server, while Vercel Functions are short-lived requests.
This repository therefore includes a separate Vercel-compatible entrypoint:

- `server.py` — a small Flask JSON API that loads the same saved model.
- `public/` — a static HTML/CSS/JS frontend served by Vercel's CDN.
- `nltk_data/` — the English stop-words and `punkt_tab` files needed by the
  shared preprocessing function.

The local Streamlit app remains available; it is not used by Vercel.

### Manual deployment from GitHub

1. Commit and push the current repository to `main`.
2. In Vercel, choose **Add New → Project** and import
   `nitish0009/sentiment-analyser`.
3. Keep the repository root as the project root.
4. Framework Preset: **Flask** (Vercel should detect it from `requirements.txt`).
5. Build Command: leave empty.
6. Output Directory: leave empty. Vercel serves `public/` at the site root and
   builds the `server.py` function automatically.
7. Install Command: leave the default (`pip install -r requirements.txt`).
8. Press **Deploy**.

If an old failed deployment is cached, open the project's **Deployments** page,
choose **Redeploy**, and enable **Clear build cache**. The original `app.py`
entrypoint error will not recur because the Streamlit file is now named
`streamlit_app.py`, while Vercel loads the top-level Flask `app` from `server.py`.

### Manual deployment with the CLI

```bash
npm install -g vercel
vercel
vercel --prod
```

The API routes are:

```text
GET  /api/health
GET  /api/metadata
POST /api/predict      {"text": "This movie was excellent."}
```

The static frontend is at `/`. The Streamlit version is still run locally with:

```bash
python -m streamlit run streamlit_app.py
```

### Test the deployment version locally

```bash
pip install -r requirements-local.txt
SERVE_STATIC=1 python -m flask --app server run --port 8000
```

Open <http://127.0.0.1:8000>. This uses the same API and static frontend that
Vercel serves.

---

## How it works

```text
Raw review
   │
   ▼
Clean text (lowercase, HTML/URL removal, contractions, letters only)
   │
   ▼
NLTK tokenisation
   │
   ▼
Stop-word removal (negations, contrast words and intensifiers preserved)
   │
   ▼
Porter stemming
   │
   ▼
TF-IDF (unigrams + bigrams, 50,000 features)
   │
   ▼
Logistic Regression → Positive / Negative + probability
```

### Why these choices?

- **TF-IDF** turns variable-length text into fixed-length numeric vectors and
  down-weights words that appear everywhere.
- **Unigrams + bigrams** capture short phrases such as `not good` and `one best`.
- **`min_df=2`, `max_df=0.95`** remove one-off typos and near-universal tokens.
- **`sublinear_tf=True`** reduces the effect of repeating the same word.
- **Naive Bayes** is a fast, strong baseline for word-count features.
- **Logistic Regression** learns a weighted linear boundary and usually handles
  overlapping vocabulary better; it won on this split.
- **Negations are kept** because removing `not` would make `good` and `not good`
  look almost identical.

---

## Reproducibility and methodology

- Split: **80% train / 20% test**, stratified, `random_state=42`.
- TF-IDF: fitted on `X_train` only, then used to `transform(X_test)`. The test
  set never influences the vocabulary or IDF weights.
- Same preprocessing function (`src/preprocessing.py`) is imported by both
  training and prediction.
- Models use fixed random seeds where applicable.
- Model selection: highest weighted F1 on the test set; accuracy is the
  tie-breaker. No hyperparameter grid search was performed.
- All metrics in the app and reports are read from
  `models/model_metadata.json`, which is written by the training run.

**Honest limitation:** because the test set is also used to choose between the
two models, the selected model's test score is slightly optimistic as an estimate
of performance on completely unseen data. A stricter design would use a separate
validation split for selection and keep the test set untouched until the final
report. This trade-off is documented in the evaluation report.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite checks:

- preprocessing behaviour, including negation preservation and empty input;
- artifact loading and vocabulary/classifier compatibility;
- prediction probability shape and confidence levels;
- metadata consistency.

Expected result: **17 tests pass**.

A quick command-line check is also available:

```bash
python src/predict.py "This movie was absolutely wonderful!" --show-cleaned
python src/predict.py "A dull, badly acted mess."
```

---

## Troubleshooting

### `LookupError: Resource punkt_tab not found`

Run:

```bash
python scripts/setup_nltk.py
chmod -R go-w ~/nltk_data
```

### `streamlit: command not found`

Use the module form:

```bash
python -m streamlit run streamlit_app.py
```

### Charts do not appear / media endpoint returns HTTP 500

This project pins **Streamlit 1.60.0** and **Starlette 0.49.3** because a newer
Starlette release broke Streamlit's static media endpoint. Keep those pins. The
app also embeds chart PNGs as base64, so the evaluation charts do not depend on
that endpoint.

### “Saved model files are missing”

Train first:

```bash
python src/train.py
```

The app loads `models/sentiment_model.pkl`, `models/tfidf_vectorizer.pkl` and
`models/model_metadata.json`. It does not train on startup.

### Training is slow on the first run

NLTK tokenisation and stemming are CPU-bound. The reference run took about
**33 seconds** to preprocess 39,665 training reviews and **8 seconds** for the
9,917 test reviews. Later runs reuse cached NLTK data but still preprocess the
text.

### `PYTHON_ENTRYPOINT_NOT_FOUND` on Vercel

The original root file was `app.py`, which was a Streamlit application. Vercel
looks for a top-level `app`, `application` or `handler` WSGI/ASGI object, and
Streamlit is not one of those servers. The deployment version now uses
`server.py` (Flask), while the Streamlit file is `streamlit_app.py`.

In the Vercel project:

1. Set the Framework Preset to **Flask**.
2. Keep the root directory at the repository root.
3. Leave the build and output directories empty.
4. Redeploy with **Clear build cache** enabled.

Vercel does not run the Streamlit app; it serves the Flask API from `server.py`
and the static frontend from `public/`. The Streamlit app remains for local use.

### Predictions look overconfident on short or sarcastic text

This is a classical bag-of-words model. It has no understanding of context,
sarcasm or negation beyond the tokens it saw in training. Use the low-confidence
warning and the Sample Reviews tricky cases to discuss this limitation.

---

## Documentation

- [Evaluation report](evaluation_report.md) — metrics, confusion matrices and
  error analysis.
- [Full project report](docs/PROJECT_REPORT.md) — 12-section project report.
- [Viva questions and answers](docs/VIVA_QA.md) — 36 likely questions.
- [Demo script](docs/DEMO_SCRIPT.md) — a 5–7 minute presentation plan.
- [Screenshot guide](docs/SCREENSHOTS.md) — optional instructions for capturing
  your own interface images.
- [Dataset notes](data/README.md) — download and CSV format.

---

## Limitations and future work

- Binary only; no neutral class.
- Sarcasm, irony and mixed opinions are hard for bag-of-words models.
- Negation is preserved but not explicitly modelled (for example, `not bad` can
  still be dominated by `bad`).
- The IMDb domain is movie reviews; product or social-media text may drift.
- Possible extensions: word-level TF-IDF with negation scope features, calibrated
  probabilities, cross-validation, a separate validation split, and comparison
  with a small transformer model as an optional advanced experiment.

---

## Dataset citation

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and
Christopher Potts. 2011. *Learning Word Vectors for Sentiment Analysis.*
Proceedings of ACL 2011.

Dataset: <https://ai.stanford.edu/~amaas/data/sentiment/>

---

## License / academic use

This repository is intended as a machine-learning/NLP project. The IMDb
dataset retains its original terms. You may reuse and adapt the code for
learning and coursework; cite the dataset and this project if you build on it.
