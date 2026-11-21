"""Vercel entrypoint for the Sentiment Analysis System.

This is the serverless companion to the Streamlit application. It exposes a
small JSON API and lets Vercel serve the static frontend from ``public/``.

Local development (also serves the frontend):

    SERVE_STATIC=1 .venv/bin/flask --app server run --port 8000

Vercel loads the top-level ``app`` object automatically.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = PROJECT_ROOT / "public"

# Use the small NLTK data bundle committed with the project. Vercel's function
# bundle does not include the developer's home directory.
os.environ.setdefault("NLTK_DATA", str(PROJECT_ROOT / "nltk_data"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LOW_CONFIDENCE_THRESHOLD  # noqa: E402
from src.predict import (  # noqa: E402
    ArtifactNotFoundError,
    PredictionError,
    SentimentPredictor,
    artifacts_exist,
    read_metadata,
)

MAX_CHARS = 5000
SERVE_STATIC = os.getenv("SERVE_STATIC", "").lower() in {"1", "true", "yes"}

app = Flask(__name__)


@lru_cache(maxsize=1)
def get_predictor() -> SentimentPredictor:
    """Load the Joblib artifacts once per warm serverless instance."""

    return SentimentPredictor()


def _chart_urls() -> list[dict[str, str]]:
    return [
        {
            "title": "Model comparison",
            "src": "/reports/model_comparison.png",
            "alt": "Accuracy, precision, recall and F1 comparison for both models",
        },
        {
            "title": "Logistic Regression confusion matrix",
            "src": "/reports/confusion_matrix_logistic_regression.png",
            "alt": "Confusion matrix for Logistic Regression on the test set",
        },
        {
            "title": "Naive Bayes confusion matrix",
            "src": "/reports/confusion_matrix_naive_bayes.png",
            "alt": "Confusion matrix for Multinomial Naive Bayes on the test set",
        },
        {
            "title": "Learned TF-IDF features",
            "src": "/reports/top_features.png",
            "alt": "Top positive and negative TF-IDF features",
        },
    ]


def _metadata_payload() -> dict:
    metadata = read_metadata()
    dataset = metadata.get("dataset", {})
    summary = dict(dataset.get("summary", {}))
    # Never expose the local developer path in a deployed API response.
    summary.pop("source_file", None)
    return {
        "project": metadata.get("project", "Sentiment Analysis System"),
        "model_name": metadata.get("model_name", "Selected model"),
        "model_key": metadata.get("model_key", ""),
        "trained_at_utc": metadata.get("trained_at_utc", ""),
        "neutral_supported": metadata.get("labels", {}).get("neutral_supported", False),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "dataset": {
            "name": dataset.get("name", ""),
            "citation": dataset.get("citation", ""),
            "summary": summary,
        },
        "split": metadata.get("split", {}),
        "preprocessing": metadata.get("preprocessing", {}),
        "tfidf": metadata.get("tfidf", {}),
        "models": metadata.get("models", {}),
        "charts": _chart_urls(),
    }


@app.get("/api/health")
def health():
    ready = artifacts_exist()
    return jsonify(
        {
            "status": "ok" if ready else "missing_artifacts",
            "model_ready": ready,
            "model_name": read_metadata().get("model_name") if ready else None,
        }
    ), (200 if ready else 503)


@app.get("/api/metadata")
def metadata():
    try:
        return jsonify(_metadata_payload())
    except ArtifactNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON body with a 'text' field."}), 400

    text = payload.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": "The 'text' field must be a string."}), 400
    if not text.strip():
        return jsonify({"error": "Please enter some text to analyse."}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"error": f"Please keep the review under {MAX_CHARS:,} characters."}), 400

    try:
        prediction = get_predictor().predict(text)
    except PredictionError as exc:
        return jsonify({"error": str(exc)}), 400
    except ArtifactNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:  # pragma: no cover - defensive server-side logging
        app.logger.exception("Unhandled prediction failure")
        return jsonify({"error": "The model could not process this review."}), 500

    result = prediction.to_dict()
    result["label"] = result["sentiment"].capitalize()
    result["confidence_percent"] = round(result["confidence"] * 100, 1)
    result["max_chars"] = MAX_CHARS
    return jsonify(result)


# ---------------------------------------------------------------------------
# Local-only static serving. On Vercel, public/ is served by the CDN.
# ---------------------------------------------------------------------------
if SERVE_STATIC:

    @app.get("/")
    def local_index():
        return send_from_directory(PUBLIC_DIR, "index.html")

    @app.get("/<path:filename>")
    def local_static(filename: str):
        candidate = (PUBLIC_DIR / filename).resolve()
        if not candidate.is_file() or PUBLIC_DIR.resolve() not in candidate.parents:
            abort(404)
        return send_from_directory(PUBLIC_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
