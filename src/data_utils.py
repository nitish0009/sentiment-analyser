"""Dataset loading, validation and cleaning.

The raw IMDb folders are converted to a CSV by ``scripts/download_dataset.py``.
This module validates that CSV and removes missing/duplicate rows *before* the
train-test split, so the test set never contains a review that also appears in
training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH, LABEL_TO_ID


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or malformed."""


@dataclass
class DatasetSummary:
    source_file: str
    rows_loaded: int
    missing_reviews: int
    missing_labels: int
    invalid_labels: int
    empty_reviews: int
    duplicates_removed: int
    final_rows: int
    positive_reviews: int
    negative_reviews: int
    average_words: float
    median_words: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_dataset(path: str | Path = DATASET_PATH) -> tuple[pd.DataFrame, DatasetSummary]:
    """Load ``dataset.csv``, validate it and remove unusable rows.

    Expected columns
    ----------------
    review    : raw review text
    sentiment : "positive" / "negative" (also accepts pos/neg and 0/1)

    Returns
    -------
    (clean_df, summary)
        ``clean_df`` has exactly two columns: ``review`` and ``sentiment``.
    """

    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at: {path}\n"
            "Create it with:\n"
            "    python scripts/download_dataset.py\n"
            "Or place a CSV with 'review' and 'sentiment' columns at that path."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DatasetError(f"Could not read the dataset CSV: {exc}") from exc

    if df.empty:
        raise DatasetError("The dataset CSV is empty.")

    # Accept slightly different capitalisation of the column names.
    df.columns = [str(col).strip().lower() for col in df.columns]
    required = {"review", "sentiment"}
    if not required.issubset(df.columns):
        raise DatasetError(
            "The dataset must contain the columns 'review' and 'sentiment'. "
            f"Found: {list(df.columns)}"
        )

    rows_loaded = len(df)
    df = df[["review", "sentiment"]].copy()

    missing_reviews = int(df["review"].isna().sum())
    missing_labels = int(df["sentiment"].isna().sum())

    # Normalise labels before dropping anything so counts are accurate.
    labels = (
        df["sentiment"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    label_map = {
        "positive": "positive",
        "pos": "positive",
        "1": "positive",
        "1.0": "positive",
        "negative": "negative",
        "neg": "negative",
        "0": "negative",
        "0.0": "negative",
    }
    normalised = labels.map(label_map)
    invalid_mask = labels.notna() & normalised.isna()
    invalid_labels = int(invalid_mask.sum())
    df["sentiment"] = normalised

    # Drop rows with no text, no label, or an unsupported label.
    df = df.dropna(subset=["review", "sentiment"])
    df["review"] = df["review"].astype(str).str.strip()
    empty_reviews = int((df["review"] == "").sum())
    df = df[df["review"] != ""]

    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["review"], keep="first")
    duplicates_removed = before_duplicates - len(df)
    df = df.reset_index(drop=True)

    if df.empty:
        raise DatasetError("No usable rows remain after cleaning the dataset.")

    if df["sentiment"].nunique() < 2:
        raise DatasetError(
            "The dataset must contain both positive and negative reviews. "
            f"Found only: {df['sentiment'].unique().tolist()}"
        )

    word_counts = df["review"].str.split().str.len()
    counts = df["sentiment"].value_counts()

    summary = DatasetSummary(
        source_file=str(path),
        rows_loaded=rows_loaded,
        missing_reviews=missing_reviews,
        missing_labels=missing_labels,
        invalid_labels=invalid_labels,
        empty_reviews=empty_reviews,
        duplicates_removed=duplicates_removed,
        final_rows=len(df),
        positive_reviews=int(counts.get("positive", 0)),
        negative_reviews=int(counts.get("negative", 0)),
        average_words=round(float(word_counts.mean()), 2),
        median_words=float(word_counts.median()),
    )

    # Keep the label encoding in one place. Training uses LABEL_TO_ID.
    unknown = set(df["sentiment"].unique()) - set(LABEL_TO_ID)
    if unknown:
        raise DatasetError(f"Unsupported labels after cleaning: {unknown}")

    return df, summary
