"""Download the IMDb 50K dataset and convert it to ``data/dataset.csv``.

Usage
-----
    python scripts/download_dataset.py

The script:
1. downloads ``aclImdb_v1.tar.gz`` from Stanford (about 84 MB),
2. extracts the 50,000 positive/negative review text files,
3. combines them into a shuffled CSV with ``review,sentiment`` columns,
4. removes the raw archive/folders unless ``--keep-raw`` is used.

The CSV is large (about 65 MB), so it is intentionally excluded from git.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATASET_PATH,
    IMDB_ARCHIVE_PATH,
    IMDB_RAW_DIR,
    IMDB_URL,
    RANDOM_STATE,
)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def download_archive(archive_path: Path, force: bool = False) -> None:
    """Download the IMDb archive with a simple progress indicator."""

    if archive_path.exists() and archive_path.stat().st_size > 1_000_000 and not force:
        print(f"Using existing archive: {archive_path}")
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if force and archive_path.exists():
        archive_path.unlink()

    print(f"Downloading IMDb dataset from:\n  {IMDB_URL}")
    print("This is about 84 MB and may take a few minutes ...")

    def report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = downloaded / total_size * 100
        if block_num % 200 == 0 or downloaded >= total_size:
            print(
                f"\r  {percent:5.1f}%  "
                f"({_human_size(downloaded)} / {_human_size(total_size)})",
                end="",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(IMDB_URL, archive_path, reporthook=report)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Download failed: {exc}\n"
            "Check your internet connection, then run the script again.\n"
            f"You can also download {IMDB_URL} manually, place it at "
            f"{archive_path}, and rerun the script."
        ) from exc
    print()


def extract_archive(archive_path: Path, raw_root: Path, force: bool = False) -> None:
    """Safely extract the IMDb tar.gz archive."""

    if raw_root.exists() and (raw_root / "train" / "pos").exists() and not force:
        print(f"Using existing extracted data: {raw_root}")
        return

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}. Run the download step first."
        )

    raw_root.parent.mkdir(parents=True, exist_ok=True)
    if force and raw_root.exists():
        shutil.rmtree(raw_root)

    print(f"Extracting {archive_path.name} ...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # The filter argument blocks unsafe paths (available in Python 3.12+;
            # the fallback keeps the script compatible with older Python).
            try:
                tar.extractall(path=raw_root.parent, filter="data")
            except TypeError:
                tar.extractall(path=raw_root.parent)
    except tarfile.TarError as exc:
        raise RuntimeError(f"Could not extract the archive: {exc}") from exc

    if not (raw_root / "train" / "pos").exists():
        raise RuntimeError(
            "Extraction finished but the expected folder "
            f"{raw_root / 'train' / 'pos'} was not created."
        )


def build_csv(raw_root: Path, output_csv: Path) -> pd.DataFrame:
    """Read all IMDb text files and write a shuffled two-column CSV."""

    rows: list[dict[str, str]] = []
    for split in ("train", "test"):
        for label in ("pos", "neg"):
            folder = raw_root / split / label
            files = sorted(folder.glob("*.txt"))
            if not files:
                raise RuntimeError(f"No review files found in {folder}")
            print(f"Reading {len(files):>5,} {label} reviews from {split} ...")
            for file_path in files:
                review = file_path.read_text(encoding="utf-8", errors="replace").strip()
                if review:
                    rows.append(
                        {
                            "review": review,
                            "sentiment": "positive" if label == "pos" else "negative",
                        }
                    )

    if not rows:
        raise RuntimeError("No reviews were read. Check the extracted dataset.")

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    counts = df["sentiment"].value_counts()
    print("\nDataset created successfully")
    print(f"  File      : {output_csv}")
    print(f"  Rows      : {len(df):,}")
    print(f"  Positive  : {counts.get('positive', 0):,}")
    print(f"  Negative  : {counts.get('negative', 0):,}")
    print(f"  Size      : {_human_size(output_csv.stat().st_size)}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_PATH,
        help="Destination CSV (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the downloaded archive and extracted text files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download, re-extract and rebuild even if files exist.",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Dataset already exists: {args.output}")
        print("Use --force to rebuild it from scratch.")
        return 0

    try:
        if not (IMDB_RAW_DIR / "train" / "pos").exists():
            download_archive(IMDB_ARCHIVE_PATH, force=args.force)
            extract_archive(IMDB_ARCHIVE_PATH, IMDB_RAW_DIR, force=args.force)
        else:
            print(f"Using existing extracted data: {IMDB_RAW_DIR}")

        build_csv(IMDB_RAW_DIR, args.output)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    if not args.keep_raw:
        print("\nCleaning up raw download files ...")
        shutil.rmtree(IMDB_RAW_DIR.parent, ignore_errors=True)
        if IMDB_ARCHIVE_PATH.exists():
            IMDB_ARCHIVE_PATH.unlink()
        print("Done. Only data/dataset.csv is kept.")
    else:
        print("\nRaw files kept because --keep-raw was used.")

    print(
        "\nNext step:\n"
        "    python src/train.py\n"
        "Then launch the app:\n"
        "    python -m streamlit run app.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
