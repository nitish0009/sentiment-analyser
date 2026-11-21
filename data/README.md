# Dataset

The project uses the **IMDb Large Movie Review Dataset** (often called IMDb 50K).

- **Source:** https://ai.stanford.edu/~amaas/data/sentiment/
- **Authors:** Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang,
  Andrew Y. Ng, Christopher Potts (ACL 2011).
- **Size:** 50,000 labelled reviews (25,000 positive + 25,000 negative).
  The archive also contains unlabelled reviews, which this project does not use.
- **Task:** binary sentiment classification.

## Automatic setup (recommended)

From the project root:

```bash
python scripts/download_dataset.py
```

The script downloads `aclImdb_v1.tar.gz` (about 84 MB), extracts the review text
files, and writes a shuffled `data/dataset.csv` with two columns:

| Column    | Meaning                                      |
|-----------|----------------------------------------------|
| review    | raw review text                              |
| sentiment | `positive` or `negative`                     |

It then removes the raw archive and extracted folders. Use `--keep-raw` if you
want to inspect them. The CSV is about 63 MB and is intentionally ignored by
git.

## Manual setup

1. Download `aclImdb_v1.tar.gz` from the Stanford link above.
2. Extract it so that the folder structure is `data/raw/aclImdb/train/pos`,
   `data/raw/aclImdb/train/neg`, etc.
3. Run `python scripts/download_dataset.py --keep-raw` — it will detect the
   existing extracted folders and build the CSV without downloading again.

## Alternative dataset

You can use any CSV with `review` and `sentiment` columns. Labels may be
`positive`/`negative` (or `pos`/`neg`, `1`/`0`). Neutral labels are not supported
by the training code in this project because the chosen dataset has no neutral
examples. See the main README for why the app does not fake a neutral class.

## Data cleaning performed by the project

`src/data_utils.py` removes missing reviews, empty reviews, unsupported labels
and exact duplicate review texts before the train-test split. On the IMDb data,
this leaves **49,582** reviews (418 exact duplicates removed). The split is
stratified 80/20 with `random_state=42`.
