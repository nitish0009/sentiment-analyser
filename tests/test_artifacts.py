"""Integration tests for the saved model and prediction wrapper.

These tests are skipped automatically if the model has not been trained yet.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (  # noqa: E402
    ArtifactNotFoundError,
    PredictionError,
    SentimentPredictor,
    artifacts_exist,
    read_metadata,
)


@unittest.skipUnless(artifacts_exist(), "Train the model first: python src/train.py")
class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = SentimentPredictor()

    def test_metadata_is_readable(self) -> None:
        metadata = read_metadata()
        self.assertIn("model_name", metadata)
        self.assertEqual(metadata["labels"]["supported_classes"], ["negative", "positive"])
        self.assertFalse(metadata["labels"]["neutral_supported"])

    def test_positive_prediction_has_valid_probabilities(self) -> None:
        result = self.predictor.predict(
            "I absolutely loved this film. The acting was brilliant and moving."
        )
        self.assertEqual(result.sentiment, "positive")
        self.assertGreater(result.confidence, 0.5)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=4)
        self.assertIn("negative", result.probabilities)
        self.assertIn("positive", result.probabilities)
        self.assertTrue(result.cleaned_text)

    def test_negative_prediction(self) -> None:
        result = self.predictor.predict(
            "A boring, badly written mess. I regret spending two hours on it."
        )
        self.assertEqual(result.sentiment, "negative")
        self.assertGreater(result.confidence, 0.5)

    def test_empty_text_raises(self) -> None:
        with self.assertRaises(PredictionError):
            self.predictor.predict("   ")

    def test_punctuation_only_text_raises(self) -> None:
        with self.assertRaises(PredictionError):
            self.predictor.predict("!!! ??? ...")

    def test_non_string_raises(self) -> None:
        with self.assertRaises(PredictionError):
            self.predictor.predict(None)  # type: ignore[arg-type]


class MissingArtifactTests(unittest.TestCase):
    def test_clear_error_for_missing_artifacts(self) -> None:
        with self.assertRaises(ArtifactNotFoundError):
            SentimentPredictor(
                model_path=PROJECT_ROOT / "does-not-exist.pkl",
                vectorizer_path=PROJECT_ROOT / "does-not-exist-2.pkl",
                metadata_path=PROJECT_ROOT / "does-not-exist-3.json",
            )


if __name__ == "__main__":
    unittest.main()
