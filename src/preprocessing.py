"""Text preprocessing shared by training and prediction.

The SAME ``preprocess_text`` function is imported by both ``train.py`` and
``predict.py``. This avoids a very common bug in student projects where the
notebook cleans text differently from the deployed application.

Pipeline
--------
1. Normalise apostrophes and expand common English contractions.
2. Unescape HTML entities and remove HTML tags.
3. Remove URLs and e-mail addresses.
4. Lowercase the text.
5. Replace every non-letter character with a space.
6. Collapse extra whitespace.
7. Tokenise with NLTK's ``word_tokenize``.
8. Remove English stop-words while KEEPING negation words such as "not".
9. Apply NLTK's Porter stemmer to reduce inflected words to a common root.
10. Join the surviving tokens back into one clean string for TF-IDF.

Why keep negations?
-------------------
"good" and "not good" have opposite meanings. Removing "not" as a stop-word
would make those two reviews look almost identical. We therefore subtract a
small set of sentiment-bearing words from NLTK's stop-word list.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from src.nltk_setup import ensure_nltk_data

# ---------------------------------------------------------------------------
# Regular expressions (compiled once for speed)
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+\b)",
    flags=re.IGNORECASE,
)
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# Contractions are expanded so that "wasn't" becomes "was not" instead of
# the meaningless tokens "was" and "t". Order matters: won't/can't/shan't
# must be handled before the generic "n't" rule.
_CONTRACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'m\b"), " am"),
)

# Words that should NOT be removed as stop-words.
# Negations prevent meaning reversal; contrast/intensifier words are useful
# features for sentiment ("not very good", "but the ending was great").
_SENTIMENT_KEEP_WORDS = {
    "not", "no", "nor", "never", "none", "nothing", "neither", "nowhere",
    "cannot", "without",
    "but", "however", "yet", "although", "though",
    "very", "too", "really", "extremely",
}

_STOP_WORDS: set[str] | None = None
_STEMMER = PorterStemmer()


class PreprocessingError(ValueError):
    """Raised when the preprocessing input cannot be processed."""


def get_stop_words() -> set[str]:
    """Load NLTK stop-words once, minus sentiment-bearing words."""

    global _STOP_WORDS
    if _STOP_WORDS is None:
        ensure_nltk_data(download=False)
        _STOP_WORDS = set(stopwords.words("english")) - _SENTIMENT_KEEP_WORDS
    return _STOP_WORDS


@lru_cache(maxsize=100_000)
def _stem(word: str) -> str:
    """Cache stemmed words; reviews repeat the same words many times."""

    return _STEMMER.stem(word)


def _expand_contractions(text: str) -> str:
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return text


def preprocess_text(text: str) -> str:
    """Clean one review and return a space-separated string of tokens.

    Parameters
    ----------
    text:
        Raw review / comment.

    Returns
    -------
    str
        Cleaned text suitable for the TF-IDF vectorizer. An empty string is
        returned if the input contains no usable words.

    Raises
    ------
    PreprocessingError
        If ``text`` is not a string (e.g. NaN from a broken CSV).
    """

    if not isinstance(text, str):
        raise PreprocessingError(
            "Review text must be a string. Missing values should be removed "
            "before preprocessing."
        )

    # 1. Normalise typographic apostrophes, lower-case, expand contractions.
    text = text.replace("’", "'").replace("‘", "'")
    text = text.lower()
    text = _expand_contractions(text)

    # 2. Decode entities such as &amp; and remove HTML tags like <br />.
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)

    # 3. Remove URLs and e-mail addresses.
    text = _URL_RE.sub(" ", text)

    # 4-6. Keep only letters/spaces, then collapse whitespace.
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    # 7. NLTK tokenisation. preserve_line=True avoids unnecessary sentence
    #    splitting for short review-like text.
    tokens = word_tokenize(text, preserve_line=True)

    # 8-9. Remove stop-words and stem the remaining alphabetic tokens.
    stop_words = get_stop_words()
    cleaned_tokens = [
        _stem(token)
        for token in tokens
        if len(token) > 1 and token.isalpha() and token not in stop_words
    ]

    # 10. TF-IDF works on whole documents; join tokens with single spaces.
    return " ".join(cleaned_tokens)


def token_count(cleaned_text: str) -> int:
    """Return the number of tokens in already-preprocessed text."""

    return len(cleaned_text.split()) if cleaned_text else 0
