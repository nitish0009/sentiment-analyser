"""Central configuration for the Sentiment Analysis System.

Keeping paths and important constants in one place means training,
prediction and the Streamlit app always use the same settings.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATASET_PATH = DATA_DIR / "dataset.csv"
IMDB_ARCHIVE_PATH = DATA_DIR / "aclImdb_v1.tar.gz"
IMDB_RAW_DIR = DATA_DIR / "raw" / "aclImdb"

MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# TF-IDF settings (saved in the metadata as well)
# ---------------------------------------------------------------------------
MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# The dataset is binary. There is deliberately no "neutral" class because the
# IMDb data does not contain neutral labels. See README.md for the rationale.
LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}

# Below this confidence the UI shows a caution message. The prediction is still
# one of the two trained classes; it is not turned into a fake "neutral" label.
LOW_CONFIDENCE_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# Dataset details (used in documentation and the UI)
# ---------------------------------------------------------------------------
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATASET_NAME = "IMDb Movie Reviews (50,000 reviews)"
DATASET_CITATION = (
    "Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, "
    "Andrew Y. Ng, and Christopher Potts. 2011. Learning Word Vectors for "
    "Sentiment Analysis. ACL 2011."
)
