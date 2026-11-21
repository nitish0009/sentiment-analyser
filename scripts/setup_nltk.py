"""Download the NLTK ``stopwords`` and ``punkt_tab`` data packages.

Run once after installing the Python requirements:

    python scripts/setup_nltk.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nltk_setup import ensure_nltk_data, missing_nltk_resources  # noqa: E402


def main() -> int:
    print("Checking NLTK data ...")
    missing = missing_nltk_resources()
    if not missing:
        print("All required NLTK resources are already installed.")
        return 0

    print(f"Missing packages: {', '.join(missing)}")
    try:
        ensure_nltk_data(download=True, quiet=False)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print(
            "\nManual alternative:\n"
            "1. Download the zip files from "
            "https://github.com/nltk/nltk_data/tree/gh-pages/packages\n"
            "   - corpora/stopwords.zip\n"
            "   - tokenizers/punkt_tab.zip\n"
            "2. Unzip them inside ~/nltk_data (keep the corpora/ and "
            "tokenizers/ folders).\n"
            "3. Run this script again."
        )
        return 1

    print("NLTK data is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
