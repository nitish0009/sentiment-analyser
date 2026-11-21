"""Small helper that makes sure the required NLTK resources exist.

The project uses only two small NLTK data packages:

* ``stopwords``  - the standard English stop-word list.
* ``punkt_tab``  - the tokenizer tables used by NLTK's ``word_tokenize``.

WordNet is not required because the project uses the faster Porter stemmer.
"""

from __future__ import annotations

import nltk

# Friendly names -> (folder inside nltk_data, resource name)
_REQUIRED_RESOURCES = {
    "stopwords": ("corpora", "stopwords"),
    "punkt_tab": ("tokenizers", "punkt_tab"),
}


class NLTKResourceError(RuntimeError):
    """Raised when NLTK data is missing and cannot be downloaded."""


def missing_nltk_resources() -> list[str]:
    """Return the names of NLTK packages that are not installed."""

    missing: list[str] = []
    for package, (folder, name) in _REQUIRED_RESOURCES.items():
        try:
            nltk.data.find(f"{folder}/{name}")
        except LookupError:
            missing.append(package)
    return missing


def ensure_nltk_data(download: bool = False, quiet: bool = True) -> None:
    """Check for the required NLTK data and optionally download it.

    Parameters
    ----------
    download:
        If True, try ``nltk.download`` for anything that is missing.
        Training passes True; the web app passes False so that a page load
        never silently waits on a large download.
    quiet:
        Suppress NLTK's downloader chatter.
    """

    missing = missing_nltk_resources()
    if not missing:
        return

    if download:
        for package in missing:
            try:
                nltk.download(package, quiet=quiet)
            except Exception as exc:  # network / SSL problems
                raise NLTKResourceError(
                    f"Could not download the NLTK package '{package}': {exc}"
                ) from exc
        missing = missing_nltk_resources()

    if missing:
        packages = " ".join(missing)
        raise NLTKResourceError(
            "Missing NLTK data: "
            f"{', '.join(missing)}.\n"
            "Install it once with:\n"
            f"    python -m nltk.downloader {packages}\n"
            "If your network blocks the NLTK downloader, run:\n"
            "    python scripts/setup_nltk.py\n"
            "or download the packages manually from "
            "https://github.com/nltk/nltk_data/tree/gh-pages/packages"
        )
