"""Unit tests for the shared preprocessing pipeline.

Run with:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import PreprocessingError, preprocess_text, token_count  # noqa: E402


class PreprocessingTests(unittest.TestCase):
    def test_output_is_lowercase_and_stemmed(self) -> None:
        cleaned = preprocess_text("I LOVED this BEAUTIFUL movie!")
        self.assertIn("love", cleaned)
        self.assertIn("beauti", cleaned)
        self.assertNotIn("LOVED", cleaned)

    def test_html_tags_and_entities_are_removed(self) -> None:
        cleaned = preprocess_text("Great film<br />with &amp; superb acting")
        self.assertNotIn("br", cleaned)
        self.assertNotIn("amp", cleaned)
        self.assertIn("great", cleaned)
        self.assertIn("superb", cleaned)

    def test_urls_and_email_addresses_are_removed(self) -> None:
        cleaned = preprocess_text(
            "Read more at https://example.com/review and email me at test@example.com"
        )
        self.assertNotIn("http", cleaned)
        self.assertNotIn("example", cleaned)
        self.assertNotIn("test", cleaned)

    def test_stopwords_are_removed(self) -> None:
        cleaned = preprocess_text("The movie was in the theatre and it was good")
        for stopword in ("the", "was", "and", "it"):
            self.assertNotIn(stopword, cleaned.split())
        self.assertIn("good", cleaned)

    def test_negations_are_preserved(self) -> None:
        cleaned = preprocess_text("This movie was not good at all")
        tokens = cleaned.split()
        self.assertIn("not", tokens)
        self.assertIn("good", tokens)
        self.assertIn("not good", cleaned)

    def test_contractions_become_negations(self) -> None:
        cleaned = preprocess_text("The film wasn't interesting and I don't recommend it")
        self.assertIn("not", cleaned.split())
        self.assertIn("interest", cleaned)

    def test_extra_whitespace_is_collapsed(self) -> None:
        cleaned = preprocess_text("   Wonderful     acting \n\n and   story   ")
        self.assertEqual(cleaned, "wonder act stori")

    def test_empty_and_punctuation_only_input(self) -> None:
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text("!!! ??? ..."), "")
        self.assertEqual(token_count(""), 0)
        self.assertEqual(token_count("great act"), 2)

    def test_non_string_raises_clear_error(self) -> None:
        with self.assertRaises(PreprocessingError):
            preprocess_text(None)  # type: ignore[arg-type]

    def test_numbers_and_special_characters_are_removed(self) -> None:
        cleaned = preprocess_text("10/10!!! Best movie ever #mustwatch @friend")
        tokens = cleaned.split()
        self.assertNotIn("10", tokens)
        self.assertNotIn("#mustwatch", tokens)
        self.assertIn("best", tokens)


if __name__ == "__main__":
    unittest.main()
