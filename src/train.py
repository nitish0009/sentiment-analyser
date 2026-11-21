"""Train, evaluate, compare and save the sentiment models.

Run from the project root:

    python src/train.py

What this script does
---------------------
1. Loads and validates data/dataset.csv.
2. Plots the class distribution and prints a dataset overview.
3. Makes a stratified 80/20 train-test split.
4. Preprocesses the text with the shared preprocessing module.
5. Fits TF-IDF **on the training split only** (no data leakage).
6. Trains Multinomial Naive Bayes and Logistic Regression.
7. Calculates accuracy, precision, recall, F1 and confusion matrices.
8. Selects the model with the best weighted F1-score.
9. Saves the model, vectorizer, metadata and evaluation charts with Joblib.

The test set is used only for final evaluation and model selection; it is
never used to fit the vectorizer or either classifier.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Allow both "python src/train.py" and "python -m src.train".
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATASET_CITATION,
    DATASET_NAME,
    DATASET_PATH,
    ID_TO_LABEL,
    LABEL_TO_ID,
    MAX_DF,
    MAX_FEATURES,
    METADATA_PATH,
    MIN_DF,
    MODEL_PATH,
    MODELS_DIR,
    NGRAM_RANGE,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    VECTORIZER_PATH,
)
from src.data_utils import DatasetError, load_dataset  # noqa: E402
from src.evaluation import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    compute_metrics,
    extract_top_features,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    save_classification_report,
)
from src.nltk_setup import NLTKResourceError, ensure_nltk_data  # noqa: E402
from src.preprocessing import preprocess_text  # noqa: E402


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _preprocess_list(texts: list[str], split_name: str) -> tuple[list[str], float]:
    """Apply the shared preprocessing function to every review."""

    print(f"Preprocessing {len(texts):,} {split_name} reviews ...")
    started = time.perf_counter()
    cleaned: list[str] = []
    for index, text in enumerate(texts, start=1):
        cleaned.append(preprocess_text(text))
        if index % 5000 == 0 or index == len(texts):
            print(f"  {index:>6,} / {len(texts):,}")
    elapsed = time.perf_counter() - started
    print(f"  done in {elapsed:.1f}s")
    return cleaned, elapsed


def _save_dataset_preview(df: pd.DataFrame, path: Path) -> None:
    """Save a few real reviews for the written report."""

    with path.open("w", encoding="utf-8") as handle:
        handle.write("DATASET SAMPLE (3 reviews per class)\n")
        handle.write("=" * 70 + "\n\n")
        for sentiment in ("positive", "negative"):
            handle.write(f"--- {sentiment.upper()} REVIEWS ---\n\n")
            subset = df[df["sentiment"] == sentiment].head(3)
            for i, row in enumerate(subset.itertuples(index=False), start=1):
                text = " ".join(str(row.review).split())
                handle.write(f"{i}. {text[:700]}{'...' if len(text) > 700 else ''}\n\n")


def _save_preprocessing_examples(
    original: list[str],
    cleaned: list[str],
    path: Path,
    count: int = 4,
) -> None:
    """Save before/after examples to make the preprocessing easy to explain."""

    with path.open("w", encoding="utf-8") as handle:
        handle.write("PREPROCESSING EXAMPLES\n")
        handle.write("=" * 70 + "\n\n")
        for i, (raw, clean) in enumerate(zip(original[:count], cleaned[:count]), start=1):
            raw_preview = " ".join(raw.split())[:500]
            clean_preview = clean[:500]
            handle.write(f"Example {i}\n")
            handle.write(f"ORIGINAL : {raw_preview}...\n")
            handle.write(f"CLEANED  : {clean_preview}...\n")
            handle.write("-" * 70 + "\n\n")


def train(args: argparse.Namespace) -> int:
    ensure_nltk_data(download=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    _banner("1. LOADING AND CLEANING THE DATASET")
    df, summary = load_dataset(args.dataset)
    total = summary.final_rows
    print(f"Dataset          : {DATASET_NAME}")
    print(f"Rows loaded      : {summary.rows_loaded:,}")
    print(f"Missing reviews  : {summary.missing_reviews:,}")
    print(f"Missing labels   : {summary.missing_labels:,}")
    print(f"Invalid labels   : {summary.invalid_labels:,}")
    print(f"Empty reviews    : {summary.empty_reviews:,}")
    print(f"Duplicates       : {summary.duplicates_removed:,}")
    print(f"Rows after clean : {summary.final_rows:,}")
    print(
        f"Positive         : {summary.positive_reviews:,} "
        f"({summary.positive_reviews / total * 100:.1f}%)"
    )
    print(
        f"Negative         : {summary.negative_reviews:,} "
        f"({summary.negative_reviews / total * 100:.1f}%)"
    )
    print(f"Average words    : {summary.average_words:.1f} per review")
    print(f"Median words     : {summary.median_words:.1f} per review")

    _save_dataset_preview(df, REPORTS_DIR / "dataset_sample.txt")
    plot_class_distribution(df, REPORTS_DIR / "class_distribution.png")
    print(f"Saved class chart: {REPORTS_DIR / 'class_distribution.png'}")

    _banner("2. TRAIN / TEST SPLIT")
    X_raw = df["review"].tolist()
    y = df["sentiment"].map(LABEL_TO_ID).to_numpy()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Training reviews : {len(X_train_raw):,}")
    print(f"Test reviews     : {len(X_test_raw):,}")
    print(
        "Training classes : "
        f"negative={int((y_train == 0).sum()):,}, "
        f"positive={int((y_train == 1).sum()):,}"
    )
    print(
        "Test classes     : "
        f"negative={int((y_test == 0).sum()):,}, "
        f"positive={int((y_test == 1).sum()):,}"
    )

    _banner("3. TEXT PREPROCESSING")
    print("Steps: lowercase -> HTML/URL removal -> contractions -> tokenise ->")
    print("       stop-word removal (negations kept) -> Porter stemming")
    X_train_clean, train_preprocess_seconds = _preprocess_list(
        X_train_raw, "training"
    )
    X_test_clean, test_preprocess_seconds = _preprocess_list(X_test_raw, "test")
    _save_preprocessing_examples(
        X_train_raw, X_train_clean, REPORTS_DIR / "preprocessing_examples.txt"
    )

    # Sanity check: a document that becomes empty cannot be classified. This is
    # extremely rare on IMDb but should be visible if a custom dataset is used.
    empty_train = sum(1 for text in X_train_clean if not text)
    empty_test = sum(1 for text in X_test_clean if not text)
    print(f"Empty after cleaning: train={empty_train}, test={empty_test}")
    if empty_train:
        raise RuntimeError(
            "Some training reviews are empty after preprocessing. "
            "Check your dataset or preprocessing settings."
        )

    _banner("4. TF-IDF FEATURE EXTRACTION")
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=args.max_features,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=False,  # text is already lower-cased by our pipeline
        token_pattern=r"(?u)\b\w+\b",
    )
    started = time.perf_counter()
    X_train = vectorizer.fit_transform(X_train_clean)  # FIT ON TRAIN ONLY
    X_test = vectorizer.transform(X_test_clean)  # transform only
    vectorize_seconds = time.perf_counter() - started
    print(f"TF-IDF train matrix: {X_train.shape[0]:,} x {X_train.shape[1]:,}")
    print(f"TF-IDF test matrix : {X_test.shape[0]:,} x {X_test.shape[1]:,}")
    print(f"Non-zero train values: {X_train.nnz:,}")
    print(f"TF-IDF fit/transform time: {vectorize_seconds:.1f}s")

    _banner("5. MODEL TRAINING AND EVALUATION")
    models = {
        "naive_bayes": MultinomialNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    }

    results: dict[str, dict] = {}
    training_seconds: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}

    for model_key, model in models.items():
        display_name = MODEL_DISPLAY_NAMES[model_key]
        print(f"\nTraining {display_name} ...")
        started = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        training_seconds[model_key] = round(elapsed, 2)

        y_pred = model.predict(X_test)
        predictions[model_key] = y_pred
        metrics = compute_metrics(y_test, y_pred)
        results[model_key] = metrics

        print(f"  Training time : {elapsed:.1f}s")
        print(f"  Accuracy      : {metrics['accuracy']:.4f}")
        print(f"  Precision (w) : {metrics['precision']:.4f}")
        print(f"  Recall (w)    : {metrics['recall']:.4f}")
        print(f"  F1-score (w)  : {metrics['f1_score']:.4f}")
        print(
            "  Confusion matrix [[TN, FP], [FN, TP]]: "
            f"{metrics['confusion_matrix']}"
        )

        save_classification_report(
            display_name,
            metrics["classification_report"],
            REPORTS_DIR / f"classification_report_{model_key}.txt",
        )
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            display_name,
            REPORTS_DIR / f"confusion_matrix_{model_key}.png",
        )

    plot_model_comparison(results, REPORTS_DIR / "model_comparison.png")

    _banner("6. SELECTING AND SAVING THE BEST MODEL")
    # Weighted F1 is the primary selection metric; accuracy breaks ties.
    best_key = max(
        results,
        key=lambda key: (results[key]["f1_score"], results[key]["accuracy"]),
    )
    best_model = models[best_key]
    best_metrics = results[best_key]
    best_name = MODEL_DISPLAY_NAMES[best_key]
    print(f"Selected model: {best_name}")
    print(
        f"Reason: highest weighted F1-score on the test set "
        f"({best_metrics['f1_score']:.4f}); "
        f"accuracy {best_metrics['accuracy']:.4f}."
    )

    top_features = extract_top_features(
        best_model,
        vectorizer,
        best_key,
        REPORTS_DIR / "top_features.png",
    )

    joblib.dump(best_model, MODEL_PATH, compress=3)
    joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)
    print(f"Saved model      : {MODEL_PATH}")
    print(f"Saved vectorizer : {VECTORIZER_PATH}")

    metadata = {
        "project": "Sentiment Analysis System",
        "model_name": best_name,
        "model_key": best_key,
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "labels": {
            "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
            "label_to_id": LABEL_TO_ID,
            "supported_classes": ["negative", "positive"],
            "neutral_supported": False,
        },
        "dataset": {
            "name": DATASET_NAME,
            "citation": DATASET_CITATION,
            "summary": summary.to_dict(),
        },
        "split": {
            "test_size": args.test_size,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_rows": len(X_train_raw),
            "test_rows": len(X_test_raw),
            "train_class_counts": {
                ID_TO_LABEL[0]: int((y_train == 0).sum()),
                ID_TO_LABEL[1]: int((y_train == 1).sum()),
            },
            "test_class_counts": {
                ID_TO_LABEL[0]: int((y_test == 0).sum()),
                ID_TO_LABEL[1]: int((y_test == 1).sum()),
            },
        },
        "preprocessing": {
            "steps": [
                "Normalise apostrophes",
                "Expand common English contractions",
                "Unescape HTML entities and remove HTML tags",
                "Remove URLs and e-mail addresses",
                "Lowercase",
                "Remove non-letter characters",
                "Collapse whitespace",
                "Tokenise with NLTK word_tokenize",
                "Remove NLTK English stop-words, preserving negations/intensifiers",
                "Apply NLTK Porter stemming",
            ],
            "stemmer": "NLTK PorterStemmer",
            "stop_words": "NLTK English stop-word list (198 words) minus sentiment keep-words",
            "negations_preserved": [
                "not", "no", "nor", "never", "none", "nothing",
                "neither", "nowhere", "cannot", "without",
            ],
        },
        "tfidf": {
            "ngram_range": list(NGRAM_RANGE),
            "max_features_requested": args.max_features,
            "n_features_learned": int(X_train.shape[1]),
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "sublinear_tf": True,
            "lowercase": False,
            "fitted_on": "training split only",
        },
        "models": {
            key: {
                "display_name": MODEL_DISPLAY_NAMES[key],
                "training_seconds": training_seconds[key],
                "metrics": results[key],
            }
            for key in models
        },
        "selected_metrics": best_metrics,
        "selection_rule": (
            "Highest weighted F1-score on the held-out test set; "
            "accuracy is used as a tie-breaker."
        ),
        "top_features": top_features,
        "timing_seconds": {
            "preprocess_train": round(train_preprocess_seconds, 2),
            "preprocess_test": round(test_preprocess_seconds, 2),
            "tfidf_fit_transform": round(vectorize_seconds, 2),
            "training": training_seconds,
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "nltk": nltk.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "notes": (
            "The dataset contains only positive and negative reviews. "
            "The model therefore does not predict a neutral class. "
            "Low-confidence predictions are flagged in the UI instead of being "
            "forced into a neutral category."
        ),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=float)

    _banner("TRAINING COMPLETE")
    print(f"Best model       : {best_name}")
    print(f"Test accuracy    : {best_metrics['accuracy']:.4f}")
    print(f"Weighted F1      : {best_metrics['f1_score']:.4f}")
    print(f"Metadata         : {METADATA_PATH}")
    print(f"Charts/reports   : {REPORTS_DIR}")
    print("\nStart the web app with:")
    print("    python -m streamlit run streamlit_app.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the review CSV (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of data used for testing (default: 0.20)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=MAX_FEATURES,
        help="Maximum TF-IDF features (default: 50000)",
    )
    args = parser.parse_args()

    if not 0.05 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.05 and 0.5")

    try:
        return train(args)
    except (DatasetError, NLTKResourceError) as exc:
        print(f"\nERROR: {exc}")
        return 1
    except Exception:
        print("\nUnexpected error during training:\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
