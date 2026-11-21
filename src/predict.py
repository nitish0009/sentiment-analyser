"""Load the saved model and predict the sentiment of new text.

Usage
-----
    python src/predict.py "This movie was absolutely wonderful!"
    python src/predict.py                 # interactive mode
    python -m src.predict "Great cast, terrible script."

The module loads:
    models/tfidf_vectorizer.pkl
    models/sentiment_model.pkl
    models/model_metadata.json

It uses the same ``preprocess_text`` function as training, which is essential
for consistent predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ID_TO_LABEL,
    LOW_CONFIDENCE_THRESHOLD,
    METADATA_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
)
from src.preprocessing import preprocess_text, token_count  # noqa: E402


class ArtifactNotFoundError(RuntimeError):
    """Raised when the model/vectorizer has not been trained yet."""


class PredictionError(ValueError):
    """Raised when the input text cannot produce a prediction."""


@dataclass
class Prediction:
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str
    input_word_count: int
    cleaned_token_count: int
    low_confidence: bool
    confidence_level: str
    model_name: str

    def to_dict(self) -> dict:
        return asdict(self)


def artifacts_exist() -> bool:
    return all(path.exists() for path in (MODEL_PATH, VECTORIZER_PATH, METADATA_PATH))


def missing_artifacts() -> list[str]:
    return [
        str(path)
        for path in (MODEL_PATH, VECTORIZER_PATH, METADATA_PATH)
        if not path.exists()
    ]


def read_metadata(path: Path = METADATA_PATH) -> dict:
    """Read the metadata JSON produced by training."""

    if not path.exists():
        raise ArtifactNotFoundError(
            f"Model metadata not found at: {path}\n"
            "Train the model first:\n"
            "    python src/train.py"
        )
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArtifactNotFoundError(
            f"Model metadata is corrupted: {exc}. Retrain the model."
        ) from exc


class SentimentPredictor:
    """A small wrapper around the saved vectorizer + classifier."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        vectorizer_path: Path = VECTORIZER_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        missing = [str(p) for p in (model_path, vectorizer_path, metadata_path) if not p.exists()]
        if missing:
            raise ArtifactNotFoundError(
                "Saved model files are missing:\n  "
                + "\n  ".join(missing)
                + "\n\nTrain the model first:\n    python src/train.py"
            )

        self.metadata = read_metadata(metadata_path)
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        except Exception as exc:
            raise ArtifactNotFoundError(
                f"Could not load the saved model files: {exc}\n"
                "Delete the files in models/ and run 'python src/train.py' again."
            ) from exc

        if not hasattr(self.model, "predict_proba"):
            raise ArtifactNotFoundError(
                "The saved model does not support probability predictions. "
                "Retrain with the provided training script."
            )

        self.model_name = str(self.metadata.get("model_name", "Sentiment model"))
        raw_mapping = self.metadata.get("labels", {}).get("id_to_label", ID_TO_LABEL)
        self.id_to_label = {int(key): str(value) for key, value in raw_mapping.items()}

        # Fail early with a clear message if someone mixes artifacts from
        # different training runs. Note: a TfidfVectorizer's n_features_in_
        # refers to its one input column, not its output vocabulary size, so
        # compare the learned vocabulary with the classifier's feature count.
        try:
            vectorizer_features = len(self.vectorizer.get_feature_names_out())
        except Exception:
            vectorizer_features = None
        model_features = getattr(self.model, "n_features_in_", None)
        if vectorizer_features != model_features:
            raise ArtifactNotFoundError(
                "The vectorizer and model were not trained together "
                f"(vectorizer features={vectorizer_features}, "
                f"model features={model_features}). Retrain the model."
            )

    def predict(self, text: str) -> Prediction:
        if not isinstance(text, str):
            raise PredictionError("Please provide the review as text.")
        if not text.strip():
            raise PredictionError("Please enter some text to analyse.")

        cleaned = preprocess_text(text)
        if not cleaned:
            raise PredictionError(
                "No usable words were found after preprocessing. "
                "Try a longer review with real words."
            )

        features = self.vectorizer.transform([cleaned])
        probabilities = np.asarray(self.model.predict_proba(features)[0], dtype=float)
        classes = np.asarray(self.model.classes_)

        probability_map: dict[str, float] = {}
        for class_id, probability in zip(classes, probabilities):
            label = self.id_to_label.get(int(class_id), str(class_id))
            probability_map[label] = float(probability)

        if not probability_map:
            raise PredictionError(
                "The model returned no class probabilities. Retrain the model."
            )

        # Normalise defensively (predict_proba already sums to 1).
        total = sum(probability_map.values())
        if total > 0:
            probability_map = {
                label: round(probability / total, 6)
                for label, probability in probability_map.items()
            }

        best_class_id = int(classes[int(np.argmax(probabilities))])
        sentiment = self.id_to_label.get(best_class_id, str(best_class_id))
        confidence = probability_map.get(sentiment, float(np.max(probabilities)))

        if confidence >= 0.75:
            confidence_level = "High"
        elif confidence >= LOW_CONFIDENCE_THRESHOLD:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        return Prediction(
            sentiment=sentiment,
            confidence=round(float(confidence), 6),
            probabilities=probability_map,
            cleaned_text=cleaned,
            input_word_count=len(text.split()),
            cleaned_token_count=token_count(cleaned),
            low_confidence=bool(confidence < LOW_CONFIDENCE_THRESHOLD),
            confidence_level=confidence_level,
            model_name=self.model_name,
        )


def _print_prediction(prediction: Prediction, show_cleaned: bool = False) -> None:
    print("\n" + "-" * 56)
    print(f"Sentiment   : {prediction.sentiment.upper()}")
    print(f"Confidence  : {prediction.confidence * 100:.2f}% ({prediction.confidence_level})")
    print(
        "Probabilities: "
        + ", ".join(
            f"{label}={probability * 100:.1f}%"
            for label, probability in sorted(prediction.probabilities.items())
        )
    )
    print(f"Model       : {prediction.model_name}")
    print(f"Tokens used : {prediction.cleaned_token_count} of {prediction.input_word_count} raw words")
    if prediction.low_confidence:
        print(
            "Note        : Low confidence. The model is binary "
            "(positive/negative), so this is not a neutral prediction."
        )
    if show_cleaned:
        print(f"Cleaned text: {prediction.cleaned_text[:400]}")
    print("-" * 56)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="Review text to analyse")
    parser.add_argument(
        "--json", action="store_true", help="Print the full prediction as JSON"
    )
    parser.add_argument(
        "--show-cleaned",
        action="store_true",
        help="Show the preprocessed text (useful for a viva demonstration)",
    )
    args = parser.parse_args()

    try:
        predictor = SentimentPredictor()
    except ArtifactNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    text = " ".join(args.text).strip()
    if text:
        try:
            prediction = predictor.predict(text)
        except PredictionError as exc:
            print(f"ERROR: {exc}")
            return 1
        if args.json:
            print(json.dumps(prediction.to_dict(), indent=2))
        else:
            _print_prediction(prediction, show_cleaned=args.show_cleaned)
        return 0

    print("Sentiment Analyzer - interactive mode")
    print("Type a review and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("Review> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            return 0
        if not text:
            continue
        try:
            prediction = predictor.predict(text)
            if args.json:
                print(json.dumps(prediction.to_dict(), indent=2))
            else:
                _print_prediction(prediction, show_cleaned=args.show_cleaned)
        except PredictionError as exc:
            print(f"ERROR: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
