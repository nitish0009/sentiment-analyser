"""Streamlit front end for the Sentiment Analysis System.

Run from the project root:

    python -m streamlit run app.py

Pages
-----
Home              : project overview, workflow and honest scope.
Sentiment Analyzer: enter text and see a prediction + confidence.
Model Performance : real test-set metrics, charts and reports.
Sample Reviews    : ready-to-run examples (including tricky cases).
"""

from __future__ import annotations

import base64
import html as html_lib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATASET_NAME,
    METADATA_PATH,
    REPORTS_DIR,
)

# The import above intentionally avoids importing private training helpers. The
# prediction module exposes the only runtime dependency we need.
from src.predict import (  # noqa: E402
    ArtifactNotFoundError,
    PredictionError,
    SentimentPredictor,
    artifacts_exist,
    read_metadata,
)
from src.nltk_setup import NLTKResourceError  # noqa: E402

st.set_page_config(
    page_title="Sentiment Analysis System",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_CHARS = 5000

SAMPLES = [
    {
        "label": "positive",
        "title": "A film worth recommending",
        "text": (
            "I absolutely loved this film. The acting was excellent, the story "
            "kept me engaged from start to finish, and the ending was deeply "
            "satisfying. I would happily watch it again."
        ),
    },
    {
        "label": "negative",
        "title": "A disappointing movie",
        "text": (
            "A complete waste of time. The plot made no sense, the acting was "
            "wooden, and I kept checking my watch waiting for it to end. "
            "I cannot recommend this to anyone."
        ),
    },
    {
        "label": "positive",
        "title": "A product that works",
        "text": (
            "An outstanding product. It arrived on time, was easy to set up, "
            "and works exactly as advertised. The build quality feels premium "
            "and the battery lasts all day. Highly recommended."
        ),
    },
    {
        "label": "negative",
        "title": "Poor product and support",
        "text": (
            "The product broke after two days. Customer support ignored my "
            "emails and the replacement arrived damaged. Very disappointing "
            "and frustrating experience."
        ),
    },
    {
        "label": "positive",
        "title": "A lovely evening out",
        "text": (
            "What a wonderful experience! The staff were friendly, the food was "
            "delicious, and the atmosphere was perfect for a relaxed evening. "
            "I will definitely be coming back."
        ),
    },
    {
        "label": "negative",
        "title": "Bad restaurant experience",
        "text": (
            "I really wanted to like this restaurant, but the food was cold, "
            "the service was rude, and the prices were far too high. "
            "I will not be returning."
        ),
    },
]

TRICKY_SAMPLES = [
    {
        "title": "Mixed review",
        "text": (
            "The cinematography was beautiful and the cast tried hard, but the "
            "script was a mess and the ending fell flat."
        ),
        "note": "Two viewpoints in one review.",
    },
    {
        "title": "Sarcasm",
        "text": (
            "Oh great, another sequel that nobody asked for. Truly a "
            "masterpiece... if you enjoy wasting money."
        ),
        "note": "Sarcasm is hard for classical models.",
    },
    {
        "title": "Genuinely mixed",
        "text": "I liked parts of it and disliked other parts.",
        "note": "Genuinely mixed; watch the confidence warning.",
    },
]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def _inject_css() -> None:
    st.markdown(
        """
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background:
            radial-gradient(1000px 500px at 100% -10%, rgba(99, 102, 241, 0.07), transparent 60%),
            linear-gradient(180deg, #f8faff 0%, #ffffff 42%);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.1rem;
        padding-bottom: 3rem;
    }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppDeployButton"],
    .stDeployButton { display: none !important; }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebarNav"] a {
        border-radius: 9px;
        padding-top: 0.45rem;
        padding-bottom: 0.45rem;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: #eef2ff;
        font-weight: 650;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] span {
        color: #4338ca;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-top: 0.5rem;
        padding: 1rem 0.1rem 0.9rem 0.1rem;
        border-top: 1px solid #e5e7eb;
    }
    .brand-mark {
        width: 40px; height: 40px; border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: #fff; font-weight: 800; font-size: 0.85rem;
        letter-spacing: -0.02em;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.25);
    }
    .brand-title { font-weight: 800; font-size: 1.02rem; color: #0f172a; line-height: 1.1; }
    .brand-sub { color: #64748b; font-size: 0.72rem; margin-top: 0.2rem; letter-spacing: 0.02em; }

    .hero {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(circle at 92% 8%, rgba(129, 140, 248, 0.42), transparent 37%),
            radial-gradient(circle at 8% 100%, rgba(20, 184, 166, 0.2), transparent 32%),
            linear-gradient(135deg, #0f172a 0%, #1e1b4b 58%, #312e81 100%);
        color: #ffffff;
        border-radius: 20px;
        padding: 2.35rem 2.5rem 2.15rem 2.5rem;
        box-shadow: 0 20px 50px rgba(30, 27, 75, 0.18);
        margin-bottom: 1.6rem;
    }
    .hero-eyebrow {
        display: inline-flex; align-items: center; gap: 0.45rem;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.13em;
        text-transform: uppercase; color: #c7d2fe;
        background: rgba(255, 255, 255, 0.09);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 999px; padding: 0.34rem 0.7rem;
        margin-bottom: 0.9rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: clamp(1.9rem, 4vw, 2.75rem);
        line-height: 1.12;
        font-weight: 800;
        letter-spacing: -0.035em;
        margin: 0 0 0.8rem 0;
        max-width: 850px;
    }
    .hero p {
        color: #dbeafe;
        font-size: 1.03rem;
        line-height: 1.65;
        max-width: 790px;
        margin: 0;
    }
    .hero-badges { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 1.25rem; }
    .hero-badge {
        font-size: 0.74rem; font-weight: 600;
        color: #e0e7ff;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 999px;
        padding: 0.3rem 0.65rem;
    }
    .hero.compact { padding: 1.7rem 2rem; margin-bottom: 1.3rem; }
    .hero.compact h1 { font-size: clamp(1.65rem, 3vw, 2.15rem); }
    .hero.compact p { font-size: 0.98rem; }

    .section-head { margin: 1.65rem 0 0.85rem 0; }
    .section-title {
        font-size: 1.28rem; font-weight: 800; color: #0f172a;
        letter-spacing: -0.02em; line-height: 1.25;
    }
    .section-sub { color: #64748b; font-size: 0.92rem; margin-top: 0.25rem; }

    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 8px 26px rgba(15, 23, 42, 0.045);
        height: 100%;
    }
    .card-title { font-size: 0.98rem; font-weight: 700; color: #0f172a; margin-bottom: 0.35rem; }
    .card-text { color: #526075; font-size: 0.88rem; line-height: 1.6; }
    .step-num {
        font-size: 0.68rem; font-weight: 800; letter-spacing: 0.12em;
        color: #6366f1; margin-bottom: 0.45rem;
    }
    .feature-chip {
        display: inline-block; margin: 0.22rem 0.28rem 0.22rem 0;
        padding: 0.3rem 0.62rem; border-radius: 7px;
        color: #3730a3; background: #eef2ff;
        font-size: 0.74rem; font-weight: 600;
    }

    .snapshot {
        background: linear-gradient(155deg, #111827 0%, #1e1b4b 100%);
        color: #fff;
        border-radius: 17px;
        padding: 1.45rem 1.5rem;
        box-shadow: 0 16px 38px rgba(30, 27, 75, 0.18);
    }
    .snapshot-kicker { font-size: 0.66rem; letter-spacing: 0.14em; color: #a5b4fc; font-weight: 800; }
    .snapshot-model { font-size: 1.22rem; font-weight: 800; margin: 0.3rem 0 1rem 0; color: #fff; }
    .snapshot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
    .snapshot-cell {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 10px; padding: 0.65rem 0.7rem;
    }
    .snapshot-cell span { display: block; color: #a5b4fc; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .snapshot-cell strong { display: block; color: #fff; font-size: 1.22rem; margin-top: 0.18rem; }
    .snapshot-foot { color: #c7d2fe; font-size: 0.73rem; margin-top: 1rem; line-height: 1.5; }
    .snapshot-code {
        background: rgba(0,0,0,0.22); color: #e0e7ff; border-radius: 8px;
        padding: 0.6rem 0.7rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem; margin-top: 0.8rem; white-space: pre-wrap;
    }

    .notice {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        color: #78350f;
        font-size: 0.86rem;
        line-height: 1.6;
        margin: 1.1rem 0 0.4rem 0;
    }
    .notice strong { color: #92400e; }
    .soft-note {
        color: #64748b; font-size: 0.8rem; line-height: 1.55;
    }

    .metric-card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1rem 1.05rem; height: 100%;
        box-shadow: 0 6px 20px rgba(15,23,42,0.035);
        border-top: 3px solid var(--accent, #6366f1);
    }
    .metric-label { color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }
    .metric-value { color: #0f172a; font-size: 1.72rem; font-weight: 800; letter-spacing: -0.03em; margin-top: 0.25rem; }
    .metric-sub { color: #94a3b8; font-size: 0.72rem; margin-top: 0.1rem; }

    .selected-banner {
        display: flex; justify-content: space-between; align-items: center; gap: 1rem;
        background: linear-gradient(120deg, #eef2ff 0%, #f5f3ff 60%, #ecfeff 100%);
        border: 1px solid #c7d2fe;
        border-radius: 15px;
        padding: 1.15rem 1.35rem;
        margin: 0.15rem 0 1rem 0;
    }
    .selected-banner h3 { margin: 0.3rem 0 0.2rem 0; font-size: 1.3rem; color: #1e1b4b; }
    .selected-banner p { margin: 0; color: #475569; font-size: 0.84rem; line-height: 1.5; }
    .pill {
        display: inline-block; padding: 0.24rem 0.55rem; border-radius: 999px;
        font-size: 0.64rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
        background: #4f46e5; color: white;
    }
    .banner-score { text-align: right; flex-shrink: 0; }
    .banner-score strong { display: block; font-size: 1.85rem; color: #312e81; letter-spacing: -0.04em; }
    .banner-score span { display: block; font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.07em; }

    .panel-label {
        color: #64748b; font-size: 0.69rem; font-weight: 800;
        letter-spacing: 0.12em; text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .result-card {
        border-radius: 17px;
        padding: 1.3rem 1.35rem 1.05rem 1.35rem;
        border: 1px solid;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
    }
    .result-positive { background: linear-gradient(150deg, #f0fdf4, #dcfce7); border-color: #86efac; }
    .result-negative { background: linear-gradient(150deg, #fff5f5, #fee2e2); border-color: #fca5a5; }
    .result-head { display: flex; align-items: center; gap: 0.85rem; }
    .result-mark {
        width: 60px; height: 60px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.35rem; font-weight: 800; letter-spacing: 0.02em;
        flex-shrink: 0;
    }
    .result-positive .result-mark { background: #bbf7d0; color: #166534; }
    .result-negative .result-mark { background: #fecaca; color: #991b1b; }
    .result-label { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }
    .result-value { font-size: 2rem; font-weight: 850; letter-spacing: -0.035em; line-height: 1.15; margin-top: 0.1rem; }
    .result-positive .result-value { color: #166534; }
    .result-negative .result-value { color: #991b1b; }
    .confidence-chip {
        margin-left: auto; text-align: right; font-size: 1.05rem; font-weight: 800;
        border-radius: 10px; padding: 0.35rem 0.55rem; white-space: nowrap;
    }
    .result-positive .confidence-chip { background: rgba(22,101,52,0.08); color: #166534; }
    .result-negative .confidence-chip { background: rgba(153,27,27,0.08); color: #991b1b; }
    .confidence-caption { color: #64748b; font-size: 0.74rem; margin: 0.7rem 0 0.9rem 0; }
    .prob-label { display: flex; justify-content: space-between; font-size: 0.76rem; color: #475569; font-weight: 600; margin: 0.55rem 0 0.25rem 0; }
    .prob-track { height: 9px; background: rgba(100, 116, 139, 0.16); border-radius: 999px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 999px; }
    .prob-fill.pos { background: linear-gradient(90deg, #4ade80, #16a34a); }
    .prob-fill.neg { background: linear-gradient(90deg, #f87171, #dc2626); }

    .empty-result {
        border: 1.5px dashed #cbd5e1;
        border-radius: 17px;
        background: rgba(255,255,255,0.62);
        padding: 2rem 1.4rem;
        text-align: center;
        min-height: 290px;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .empty-rule { width: 48px; height: 3px; border-radius: 999px; background: #c7d2fe; margin-bottom: 0.9rem; }
    .empty-result h3 { margin: 0.6rem 0 0.3rem 0; font-size: 1.05rem; color: #1e293b; }
    .empty-result p { margin: 0; color: #64748b; font-size: 0.84rem; line-height: 1.55; max-width: 300px; }

    .token-chip {
        display: inline-block; padding: 0.27rem 0.55rem; border-radius: 7px;
        margin: 0.16rem; font-size: 0.77rem; font-weight: 600;
    }
    .token-pos { background: #dcfce7; color: #166534; }
    .token-neg { background: #fee2e2; color: #991b1b; }

    .chart-wrap {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.45rem;
        margin: 0.2rem 0 0.5rem 0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }
    .chart-wrap img {
        display: block;
        width: 100%;
        height: auto;
        border-radius: 10px;
    }

    [data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 0.9rem 1rem !important;
        font-size: 0.96rem !important;
        line-height: 1.65 !important;
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.025);
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.13) !important;
    }

    .sample-tag {
        display: inline-block; border-radius: 999px; padding: 0.22rem 0.55rem;
        font-size: 0.62rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
    }
    .sample-tag.positive { background: #dcfce7; color: #166534; }
    .sample-tag.negative { background: #fee2e2; color: #991b1b; }
    .sample-tag.tricky { background: #fef3c7; color: #92400e; }
    .sample-quote {
        color: #1e293b; font-size: 0.9rem; line-height: 1.62;
        margin: 0.6rem 0 0.55rem 0; font-style: italic;
    }
    .sample-note { color: #94a3b8; font-size: 0.72rem; margin-top: 0.2rem; }

    .sidebar-footer { color: #94a3b8; font-size: 0.72rem; line-height: 1.55; padding-top: 0.4rem; }
    .status-ready {
        display: inline-flex; align-items: center; gap: 0.4rem;
        color: #166534; background: #f0fdf4; border: 1px solid #bbf7d0;
        border-radius: 999px; padding: 0.27rem 0.6rem;
        font-size: 0.72rem; font-weight: 700;
    }
    .status-warn {
        display: inline-flex; align-items: center; gap: 0.4rem;
        color: #92400e; background: #fffbeb; border: 1px solid #fde68a;
        border-radius: 999px; padding: 0.27rem 0.6rem;
        font-size: 0.72rem; font-weight: 700;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; display: inline-block; }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 650;
        border-color: #cbd5e1;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover { border-color: #6366f1; color: #4338ca; }
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        border: none;
        color: #ffffff;
        box-shadow: 0 7px 18px rgba(79, 70, 229, 0.24);
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #4338ca, #6d28d9);
        color: #ffffff;
    }

    .footer {
        margin-top: 2.2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.76rem;
        line-height: 1.5;
    }
    .footer strong { color: #64748b; }

    @media (max-width: 700px) {
        .hero { padding: 1.6rem 1.2rem; }
        .hero h1 { font-size: 1.7rem; }
        .selected-banner { flex-direction: column; align-items: flex-start; }
        .banner-score { text-align: left; }
        .result-head { flex-wrap: wrap; }
        .confidence-chip { margin-left: 0; }
    }
</style>
""",
        unsafe_allow_html=True,
    )


def _footer() -> None:
    st.markdown(
        """
<div class="footer">
    <strong>Sentiment Analysis System</strong> · Binary sentiment classification with classical NLP<br>
    Built with Python, scikit-learn, NLTK, Pandas, Matplotlib, Seaborn, Joblib and Streamlit.
    Binary classifier trained on the IMDb 50K movie-review dataset. No paid APIs or pretrained language models.
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------
def _esc(value: object) -> str:
    return html_lib.escape(str(value))


def _hero(
    title: str,
    subtitle: str,
    eyebrow: str,
    badges: list[str] | None = None,
    compact: bool = False,
) -> str:
    badge_html = ""
    if badges:
        badge_html = (
            '<div class="hero-badges">'
            + "".join(f'<span class="hero-badge">{_esc(badge)}</span>' for badge in badges)
            + "</div>"
        )
    compact_class = " compact" if compact else ""
    return f"""
<div class="hero{compact_class}">
    <div class="hero-eyebrow">{_esc(eyebrow)}</div>
    <h1>{_esc(title)}</h1>
    <p>{_esc(subtitle)}</p>
    {badge_html}
</div>
"""


def _section(title: str, subtitle: str = "") -> None:
    sub = f'<div class="section-sub">{_esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-head"><div class="section-title">{_esc(title)}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, sublabel: str, accent: str) -> None:
    st.markdown(
        f"""
<div class="metric-card" style="--accent:{_esc(accent)}">
    <div class="metric-label">{_esc(label)}</div>
    <div class="metric-value">{_esc(value)}</div>
    <div class="metric-sub">{_esc(sublabel)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _load_metadata() -> tuple[dict | None, str | None]:
    if not METADATA_PATH.exists():
        return None, "No trained model metadata found."
    try:
        return read_metadata(), None
    except ArtifactNotFoundError as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def _load_predictor() -> SentimentPredictor:
    return SentimentPredictor()


def _get_predictor() -> tuple[SentimentPredictor | None, str | None]:
    if not artifacts_exist():
        return None, (
            "Saved model files are missing. Train the model first with:\n\n"
            "    python scripts/download_dataset.py\n"
            "    python src/train.py"
        )
    try:
        return _load_predictor(), None
    except ArtifactNotFoundError as exc:
        return None, str(exc)


def _show_image(path: Path, alt: str) -> None:
    """Embed a PNG chart directly in the page.

    The images are small (under 100 KB), so base64 embedding keeps the app
    self-contained and avoids depending on Streamlit's temporary media server.
    """

    if not path.exists():
        st.info(f"Chart not found: {path.name}. Re-run training to regenerate it.")
        return
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        st.warning(f"Could not read chart {path.name}: {exc}")
        return
    st.markdown(
        f'<div class="chart-wrap"><img src="data:image/png;base64,{encoded}" '
        f'alt="{_esc(alt)}"></div>',
        unsafe_allow_html=True,
    )


def _probability_bar(label: str, value: float, css_class: str) -> str:
    percentage = max(0.0, min(100.0, value))
    return f"""
<div class="prob-label"><span>{_esc(label)}</span><span>{percentage:.1f}%</span></div>
<div class="prob-track"><div class="prob-fill {css_class}" style="width:{percentage:.2f}%"></div></div>
"""


def _result_card(prediction: dict) -> None:
    sentiment = str(prediction.get("sentiment", "positive")).lower()
    is_positive = sentiment == "positive"
    label = "Positive" if is_positive else "Negative"
    mark = "P" if is_positive else "N"
    css = "positive" if is_positive else "negative"
    confidence = float(prediction.get("confidence", 0.0))
    level = str(prediction.get("confidence_level", ""))
    probabilities = prediction.get("probabilities", {})
    negative = float(probabilities.get("negative", 0.0)) * 100
    positive = float(probabilities.get("positive", 0.0)) * 100

    st.markdown(
        f"""
<div class="result-card result-{css}">
    <div class="result-head">
        <div class="result-mark">{mark}</div>
        <div>
            <div class="result-label">Predicted sentiment</div>
            <div class="result-value">{label}</div>
        </div>
        <div class="confidence-chip">{confidence * 100:.1f}%</div>
    </div>
    <div class="confidence-caption">
        Confidence level: <strong>{_esc(level)}</strong> ·
        This is the model's probability for the predicted class.
    </div>
    {_probability_bar("Negative probability", negative, "neg")}
    {_probability_bar("Positive probability", positive, "pos")}
</div>
""",
        unsafe_allow_html=True,
    )


def _empty_result() -> None:
    st.markdown(
        """
<div class="empty-result">
    <div class="empty-rule"></div>
    <h3>Your prediction will appear here</h3>
    <p>Type a review on the left and press <strong>Analyze sentiment</strong>.
    You will see the predicted class, confidence score and probability split.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _missing_model_notice(error: str | None = None) -> None:
    st.error(
        "The sentiment model has not been trained yet, or its saved files are missing."
    )
    if error:
        st.caption(error)
    st.markdown("Run these commands from the project root:")
    st.code(
        "python scripts/download_dataset.py\npython src/train.py\npython -m streamlit run app.py",
        language="bash",
    )
    st.info(
        "Training creates models/sentiment_model.pkl, models/tfidf_vectorizer.pkl "
        "and models/model_metadata.json. Restart or reload the app afterwards."
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar_bottom() -> None:
    metadata, _ = _load_metadata()
    with st.sidebar:
        st.markdown(
            """
<div class="brand">
    <div class="brand-mark">SA</div>
    <div>
        <div class="brand-title">Sentiment Analysis</div>
        <div class="brand-sub">Classical NLP · ML</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if artifacts_exist():
            st.markdown(
                '<div class="status-ready"><span class="status-dot"></span>Model ready</div>',
                unsafe_allow_html=True,
            )
            if metadata:
                accuracy = metadata.get("selected_metrics", {}).get("accuracy")
                if accuracy is not None:
                    st.caption(
                        f"{metadata.get('model_name', 'Model')} · "
                        f"{accuracy * 100:.2f}% test accuracy"
                    )
        else:
            st.markdown(
                '<div class="status-warn"><span class="status-dot"></span>Model not trained</div>',
                unsafe_allow_html=True,
            )

        if st.button("Reload saved model", width="stretch"):
            _load_predictor.clear()
            st.toast("Saved model reloaded from disk.")

        st.divider()
        st.markdown(
            """
<div class="sidebar-footer">
    <strong>Binary classification</strong><br>
    Positive / Negative only. The IMDb training data has no neutral labels,
    so the app does not invent a neutral class.<br><br>
    <strong>Privacy</strong><br>
    Reviews are processed in memory and are not stored or sent to any API.
</div>
""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
def render_home() -> None:
    metadata, _ = _load_metadata()
    st.markdown(
        _hero(
            "Sentiment Analysis System",
            "A classical NLP and machine-learning application that reads a review and predicts "
            "whether it is positive or negative, with a transparent confidence score.",
            "Classical NLP · Machine Learning",
            [
                "Python",
                "scikit-learn",
                "NLTK",
                "TF-IDF",
                "Streamlit",
                "No LLMs",
            ],
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        _section("What is sentiment analysis?")
        st.markdown(
            """
Sentiment analysis is the task of automatically identifying the opinion or emotion
expressed in text. A product review, a movie comment and a social-media post all
contain clues — words such as *excellent*, *boring*, *loved* or *waste* — that a
machine-learning model can learn to recognise.

This project treats the problem as **supervised binary text classification**:
the model is trained on 50,000 IMDb movie reviews that are already labelled
positive or negative, learns which words and word pairs are associated with each
class, and then predicts the label of a new, unseen review.
"""
        )
        st.markdown(
            '<div style="height:0.35rem"></div>'
            '<span class="feature-chip">Text cleaning</span>'
            '<span class="feature-chip">Tokenisation</span>'
            '<span class="feature-chip">Stop-word removal</span>'
            '<span class="feature-chip">Stemming</span>'
            '<span class="feature-chip">TF-IDF uni + bigrams</span>'
            '<span class="feature-chip">Naive Bayes</span>'
            '<span class="feature-chip">Logistic Regression</span>'
            '<span class="feature-chip">Model evaluation</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="soft-note" style="margin-top:0.8rem">The complete pipeline — dataset '
            'loading, preprocessing, training, evaluation and prediction — is implemented in '
            'readable Python modules rather than hidden inside a notebook.</p>',
            unsafe_allow_html=True,
        )

    with right:
        if metadata:
            metrics = metadata.get("selected_metrics", {})
            dataset_summary = metadata.get("dataset", {}).get("summary", {})
            split = metadata.get("split", {})
            accuracy = metrics.get("accuracy", 0) * 100
            f1 = metrics.get("f1_score", 0) * 100
            features = metadata.get("tfidf", {}).get("n_features_learned", "—")
            st.markdown(
                f"""
<div class="snapshot">
    <div class="snapshot-kicker">MODEL SNAPSHOT</div>
    <div class="snapshot-model">{_esc(metadata.get('model_name', 'Trained model'))}</div>
    <div class="snapshot-grid">
        <div class="snapshot-cell"><span>Test accuracy</span><strong>{accuracy:.2f}%</strong></div>
        <div class="snapshot-cell"><span>Weighted F1</span><strong>{f1:.2f}%</strong></div>
        <div class="snapshot-cell"><span>TF-IDF features</span><strong>{int(features):,}</strong></div>
        <div class="snapshot-cell"><span>Classes</span><strong>2</strong></div>
    </div>
    <div class="snapshot-foot">
        Trained on {split.get('train_rows', 0):,} reviews · tested on
        {split.get('test_rows', 0):,} held-out reviews.<br>
        Dataset: {int(dataset_summary.get('final_rows', 0)):,} cleaned IMDb reviews.
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="snapshot">
    <div class="snapshot-kicker">MODEL SNAPSHOT</div>
    <div class="snapshot-model">Not trained yet</div>
    <div class="snapshot-foot">Build the dataset and train both models to populate
    this panel with real, measured results.</div>
    <div class="snapshot-code">python scripts/download_dataset.py
python src/train.py</div>
</div>
""",
                unsafe_allow_html=True,
            )

    _section(
        "How the pipeline works",
        "Each stage is implemented in its own module so it can be explained and tested separately.",
    )
    steps = [
        ("01", "Load & inspect", "Read the IMDb CSV, check labels, remove missing and duplicate reviews, and visualise the class balance."),
        ("02", "Clean & tokenise", "Lowercase, remove HTML/URLs, expand contractions, tokenise with NLTK, remove stop-words and stem."),
        ("03", "TF-IDF features", "Convert clean text into weighted unigram and bigram features fitted only on the training split."),
        ("04", "Train & evaluate", "Compare Multinomial Naive Bayes with Logistic Regression using accuracy, precision, recall, F1 and confusion matrices."),
    ]
    cols = st.columns(4, gap="medium")
    for col, (num, title, description) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
<div class="card">
    <div class="step-num">STEP {num}</div>
    <div class="card-title">{_esc(title)}</div>
    <div class="card-text">{_esc(description)}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    _section("How to use the application")
    use_left, use_right = st.columns([1.5, 1], gap="large")
    with use_left:
        st.markdown(
            """
1. Open **Sentiment Analyzer** from the sidebar.
2. Paste a movie review, product review or social-media comment (up to 5,000 characters).
3. Press **Analyze sentiment**.
4. Read the predicted class, confidence percentage and per-class probabilities.
5. Use **Model Performance** to see the real test-set metrics and charts.
6. Use **Sample Reviews** for ready-made positive, negative and tricky examples.
"""
        )
    with use_right:
        st.markdown(
            """
<div class="card">
    <div class="card-title">What makes a good demo?</div>
    <div class="card-text">
        Try a clearly positive review and a clearly negative one first. Then try a mixed
        or sarcastic review to show that the model reports a binary decision and a
        confidence score rather than pretending to understand every nuance.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    cta1, cta2, cta3 = st.columns([1, 1, 2])
    with cta1:
        if st.button("Try the analyzer →", type="primary", width="stretch"):
            st.switch_page(ANALYZER_PAGE)
    with cta2:
        if st.button("View model results", width="stretch"):
            st.switch_page(PERFORMANCE_PAGE)

    st.markdown(
        """
<div class="notice">
    <strong>Honest scope:</strong> the training dataset contains only positive and negative
    reviews, so a reliable third “neutral” class is not supported. The model always returns
    one of its two trained classes. When its confidence is below 60%, the interface clearly
    flags the prediction as low-confidence instead of inventing a neutral label.
</div>
""",
        unsafe_allow_html=True,
    )
    _footer()


# ---------------------------------------------------------------------------
# Analyzer page
# ---------------------------------------------------------------------------
def _store_prediction(review: str) -> None:
    predictor, error = _get_predictor()
    if error or predictor is None:
        st.session_state.pop("last_prediction", None)
        st.error(error or "The model could not be loaded.")
        return

    try:
        with st.spinner("Cleaning text and running the saved model..."):
            prediction = predictor.predict(review)
        st.session_state["last_prediction"] = prediction.to_dict()
        st.session_state["analyzed_text"] = review
    except PredictionError as exc:
        st.session_state.pop("last_prediction", None)
        st.warning(str(exc))
    except NLTKResourceError as exc:
        st.session_state.pop("last_prediction", None)
        st.error(str(exc))
    except Exception as exc:  # defensive: keep the UI alive, show details
        st.session_state.pop("last_prediction", None)
        st.error(f"Unexpected prediction error: {exc}")
        with st.expander("Technical details"):
            st.exception(exc)


def render_analyzer() -> None:
    # A sample button can place text here before the text_area widget is created.
    if "pending_review" in st.session_state:
        st.session_state["review_text"] = st.session_state.pop("pending_review")
    auto_analyze = bool(st.session_state.pop("pending_auto_analyze", False))

    st.markdown(
        _hero(
            "Sentiment Analyzer",
            "Enter a review or comment. The saved TF-IDF model cleans the text with the "
            "exact same pipeline used during training, then predicts positive or negative.",
            "Live prediction",
            ["TF-IDF", "Logistic Regression", "Probability output"],
            compact=True,
        ),
        unsafe_allow_html=True,
    )

    if not artifacts_exist():
        _missing_model_notice()

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown('<div class="panel-label">Your text</div>', unsafe_allow_html=True)
        review = st.text_area(
            "Review or comment",
            key="review_text",
            height=230,
            max_chars=MAX_CHARS,
            placeholder=(
                "Example: The acting was brilliant and the story kept me engaged "
                "from beginning to end..."
            ),
            label_visibility="collapsed",
        )
        review = review or ""

        button_col, clear_col = st.columns([2.1, 1])
        with button_col:
            analyze_clicked = st.button(
                "Analyze sentiment",
                type="primary",
                width="stretch",
                disabled=not artifacts_exist(),
            )
        with clear_col:
            clear_clicked = st.button("Clear", width="stretch")

        st.caption(f"{len(review):,} / {MAX_CHARS:,} characters · longer reviews work, but short clear opinions are easiest to show.")

        with st.expander("Try an example instead"):
            example_choice = st.selectbox(
                "Choose a sample review",
                [f"{sample['label'].title()} · {sample['title']}" for sample in SAMPLES],
                label_visibility="collapsed",
            )
            if st.button("Load and analyze", key="load_analyzer_example", width="stretch"):
                index = [f"{s['label'].title()} · {s['title']}" for s in SAMPLES].index(example_choice)
                st.session_state["pending_review"] = SAMPLES[index]["text"]
                st.session_state["pending_auto_analyze"] = True
                st.rerun()

    if clear_clicked:
        st.session_state["pending_review"] = ""
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("analyzed_text", None)
        st.rerun()

    if analyze_clicked or auto_analyze:
        if not review.strip():
            st.warning("Please enter a review before analyzing.")
        else:
            _store_prediction(review)

    with right:
        st.markdown('<div class="panel-label">Model output</div>', unsafe_allow_html=True)
        prediction = st.session_state.get("last_prediction")
        analyzed_text = st.session_state.get("analyzed_text", "")

        if prediction:
            if analyzed_text.strip() != review.strip():
                st.info("You edited the text after this prediction. Analyze again to refresh the result.")
            _result_card(prediction)

            if prediction.get("low_confidence"):
                st.warning(
                    "Low confidence (below 60%). The model is unsure between positive and "
                    "negative. It still does not predict “neutral” because neutral labels "
                    "were not part of the training data."
                )

            with st.expander("See preprocessing details"):
                st.markdown(
                    f"**Model:** {prediction.get('model_name', 'Saved model')}  \n"
                    f"**Raw words:** {prediction.get('input_word_count', 0):,}  \n"
                    f"**Tokens after cleaning:** {prediction.get('cleaned_token_count', 0):,}"
                )
                st.caption(
                    "This is the exact token string passed to TF-IDF. Stemming can make "
                    "words look unusual (for example, “beautiful” → “beauti”)."
                )
                st.code(prediction.get("cleaned_text", ""), language=None)

            st.caption(
                "Confidence is the model's estimated probability, not a guarantee. "
                "Use the Sample Reviews page to test known positive and negative cases."
            )
        else:
            _empty_result()
            st.markdown(
                '<p class="soft-note" style="margin-top:0.8rem">The model does not receive '
                "raw text directly: preprocessing and TF-IDF convert it into numbers first.</p>",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
<div class="notice">
    <strong>Why no “neutral” result?</strong> IMDb reviews are labelled only positive or
    negative. Adding a neutral class would require neutral training data and a separately
    evaluated model. This project reports the two supported classes and exposes confidence
    honestly.
</div>
""",
        unsafe_allow_html=True,
    )
    _footer()


# ---------------------------------------------------------------------------
# Performance page
# ---------------------------------------------------------------------------
def _report_dataframe(report: dict) -> pd.DataFrame:
    df = pd.DataFrame(report).transpose()
    keep = [index for index in ["Negative", "Positive", "macro avg", "weighted avg"] if index in df.index]
    df = df.loc[keep]
    for column in ("precision", "recall", "f1-score", "support"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-score",
            "support": "Support",
        }
    )
    return df.reset_index().rename(columns={"index": "Class"})


def render_performance() -> None:
    metadata, meta_error = _load_metadata()
    if not metadata:
        st.markdown(
            _hero(
                "Model Performance",
                "The evaluation dashboard appears after the model has been trained.",
                "Evaluation",
                compact=True,
            ),
            unsafe_allow_html=True,
        )
        _missing_model_notice(meta_error)
        _footer()
        return

    selected_name = metadata.get("model_name", "Selected model")
    selected_key = metadata.get("model_key", "")
    selected_metrics = metadata.get("selected_metrics", {})
    models = metadata.get("models", {})
    split = metadata.get("split", {})
    tfidf = metadata.get("tfidf", {})
    dataset = metadata.get("dataset", {})
    summary = dataset.get("summary", {})

    st.markdown(
        _hero(
            "Model Performance",
            "Every number below was measured on a stratified, held-out 20% test split. "
            "The test set was not used to fit the vectorizer or either classifier.",
            "Evaluation dashboard",
            ["Accuracy", "Precision", "Recall", "F1", "Confusion matrix"],
            compact=True,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="selected-banner">
    <div>
        <span class="pill">Selected model</span>
        <h3>{_esc(selected_name)}</h3>
        <p>Selected automatically by the highest weighted F1-score on the test set
        (accuracy is used as a tie-breaker). The comparison is therefore based on measured
        performance, not a manual choice.</p>
    </div>
    <div class="banner-score">
        <strong>{selected_metrics.get('accuracy', 0) * 100:.2f}%</strong>
        <span>Test accuracy</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4, gap="medium")
    metric_specs = [
        ("Accuracy", "accuracy", "Overall correct predictions", "#6366f1"),
        ("Precision", "precision", "Weighted average", "#0ea5e9"),
        ("Recall", "recall", "Weighted average", "#14b8a6"),
        ("F1-score", "f1_score", "Weighted average", "#8b5cf6"),
    ]
    for col, (label, key, sublabel, accent) in zip(metric_cols, metric_specs):
        with col:
            _metric_card(
                label,
                f"{selected_metrics.get(key, 0) * 100:.2f}%",
                sublabel,
                accent,
            )

    st.caption(
        f"Measured on {split.get('test_rows', 0):,} test reviews. Precision, recall and "
        "F1 are weighted by class support; the full per-class report is shown further down."
    )

    _section(
        "Naive Bayes vs Logistic Regression",
        "Both models use the same train/test split, the same preprocessed text and the same TF-IDF matrix.",
    )
    chart_path = REPORTS_DIR / "model_comparison.png"
    _show_image(chart_path, "Naive Bayes and Logistic Regression metric comparison")

    model_rows = []
    for key, payload in models.items():
        metrics = payload.get("metrics", {})
        model_rows.append(
            {
                "Model": payload.get("display_name", key),
                "Accuracy": metrics.get("accuracy", 0),
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "F1": metrics.get("f1_score", 0),
                "Selected": "Yes" if key == selected_key else "",
            }
        )
    comparison_df = pd.DataFrame(model_rows)
    st.dataframe(
        comparison_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Precision": st.column_config.NumberColumn(format="%.4f"),
            "Recall": st.column_config.NumberColumn(format="%.4f"),
            "F1": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.markdown(
        '<p class="soft-note">Scores are fractions between 0 and 1. '
        "Logistic Regression makes fewer mistakes on this split, especially on "
        "reviews where sentiment is expressed indirectly.</p>",
        unsafe_allow_html=True,
    )

    _section(
        "Confusion matrices",
        "Rows are the true labels; columns are what the model predicted. Diagonal cells are correct.",
    )
    matrix_cols = st.columns(2, gap="large")
    ordered_keys = [selected_key] + [key for key in models if key != selected_key]
    for col, key in zip(matrix_cols, ordered_keys):
        payload = models.get(key, {})
        display_name = payload.get("display_name", key)
        matrix_path = REPORTS_DIR / f"confusion_matrix_{key}.png"
        with col:
            st.markdown(
                f"**{_esc(display_name)}**"
                + (" · selected" if key == selected_key else ""),
            )
            if matrix_path.exists():
                _show_image(matrix_path, f"Confusion matrix for {display_name}")
            else:
                st.info("Confusion-matrix image not found.")
            cm = payload.get("metrics", {}).get("confusion_matrix")
            if cm:
                st.caption(
                    f"TN={cm[0][0]:,} · FP={cm[0][1]:,} · FN={cm[1][0]:,} · TP={cm[1][1]:,}"
                )

    _section(
        "Classification reports",
        "Per-class detail: how well the model finds positive reviews and negative reviews separately.",
    )
    report_tabs = st.tabs(
        [models[key].get("display_name", key) for key in ordered_keys if key in models]
    )
    for tab, key in zip(report_tabs, ordered_keys):
        payload = models[key]
        with tab:
            report = payload.get("metrics", {}).get("classification_report", {})
            if report:
                st.dataframe(
                    _report_dataframe(report),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Precision": st.column_config.NumberColumn(format="%.4f"),
                        "Recall": st.column_config.NumberColumn(format="%.4f"),
                        "F1-score": st.column_config.NumberColumn(format="%.4f"),
                        "Support": st.column_config.NumberColumn(format="%d"),
                    },
                )
                st.caption(
                    "Support = number of test reviews in that class. Macro average gives both "
                    "classes equal weight; weighted average accounts for the number of reviews."
                )
            else:
                st.info("Classification report not found in the metadata.")

    top_features = metadata.get("top_features")
    if top_features:
        _section(
            "What the model learned",
            "The strongest global TF-IDF features for each class. These are model-level signals, not a per-review explanation.",
        )
        feature_path = REPORTS_DIR / "top_features.png"
        if feature_path.exists():
            _show_image(feature_path, "Strongest positive and negative TF-IDF features")

        chip_cols = st.columns(2, gap="large")
        with chip_cols[0]:
            st.markdown("**Strongest positive tokens**")
            pos_tokens = "".join(
                f'<span class="token-chip token-pos">{_esc(item.get("token"))}</span>'
                for item in top_features.get("positive", [])[:12]
            )
            st.markdown(pos_tokens, unsafe_allow_html=True)
        with chip_cols[1]:
            st.markdown("**Strongest negative tokens**")
            neg_tokens = "".join(
                f'<span class="token-chip token-neg">{_esc(item.get("token"))}</span>'
                for item in top_features.get("negative", [])[:12]
            )
            st.markdown(neg_tokens, unsafe_allow_html=True)
        st.markdown(
            '<p class="soft-note">Tokens are shown after stemming, so some are stems '
            '(for example, “stori” rather than “story”).</p>',
            unsafe_allow_html=True,
        )

    _section("Dataset and method", "The information a professor is likely to ask about.")
    method_cols = st.columns(3, gap="medium")
    with method_cols[0]:
        st.markdown(
            f"""
<div class="card">
    <div class="card-title">Dataset</div>
    <div class="card-text">
        {_esc(dataset.get('name', DATASET_NAME))}<br><br>
        Raw rows: <strong>{summary.get('rows_loaded', 0):,}</strong><br>
        Duplicates removed: <strong>{summary.get('duplicates_removed', 0):,}</strong><br>
        Clean rows: <strong>{summary.get('final_rows', 0):,}</strong><br>
        Positive / negative: <strong>{summary.get('positive_reviews', 0):,} / {summary.get('negative_reviews', 0):,}</strong><br>
        Average review length: <strong>{summary.get('average_words', 0):.1f} words</strong>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with method_cols[1]:
        st.markdown(
            f"""
<div class="card">
    <div class="card-title">Split & features</div>
    <div class="card-text">
        Train / test: <strong>{split.get('train_rows', 0):,} / {split.get('test_rows', 0):,}</strong><br>
        Test fraction: <strong>{split.get('test_size', 0.2) * 100:.0f}%</strong><br>
        Stratified: <strong>yes</strong><br>
        Random state: <strong>{split.get('random_state', 42)}</strong><br>
        TF-IDF n-grams: <strong>{tuple(tfidf.get('ngram_range', [1, 2]))}</strong><br>
        Features learned: <strong>{tfidf.get('n_features_learned', 0):,}</strong><br>
        Vectorizer fitted on: <strong>training split only</strong>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with method_cols[2]:
        st.markdown(
            f"""
<div class="card">
    <div class="card-title">Reproduce these results</div>
    <div class="card-text">
        The split is deterministic, so re-running training should reproduce the same
        numbers (up to tiny library-level variations).<br><br>
        <code>python src/train.py</code><br>
        <code>python -m streamlit run app.py</code><br><br>
        Total training time recorded: preprocess
        <strong>{metadata.get('timing_seconds', {}).get('preprocess_train', 0):.1f}s</strong>,
        vectorise <strong>{metadata.get('timing_seconds', {}).get('tfidf_fit_transform', 0):.1f}s</strong>.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("What do accuracy, precision, recall and F1 mean?"):
        st.markdown(
            """
- **Accuracy:** out of all test reviews, the percentage the model classified correctly.
- **Precision:** when the model predicts a class, how often it is correct. High precision means few false alarms.
- **Recall:** out of all reviews that truly belong to a class, how many the model found. High recall means few missed cases.
- **F1-score:** the harmonic mean of precision and recall; a single balanced summary.
- **Confusion matrix:** a table of true vs predicted labels. The diagonal is correct; the off-diagonal cells are the two kinds of mistakes (false positives and false negatives).

Because the test classes are nearly balanced, weighted and macro averages are almost
identical here. The classification report exposes both, plus the positive-class and
negative-class scores, so the result is not hidden behind one headline number.
"""
        )
    _footer()


# ---------------------------------------------------------------------------
# Sample reviews page
# ---------------------------------------------------------------------------
def _load_sample(text: str) -> None:
    st.session_state["pending_review"] = text
    st.session_state["pending_auto_analyze"] = True
    st.switch_page(ANALYZER_PAGE)


def render_samples() -> None:
    st.markdown(
        _hero(
            "Sample Reviews",
            "Use these ready-to-run examples during a demonstration. The first six have a "
            "clear expected sentiment; the tricky cases show the limits of a classical binary model.",
            "Demo playground",
            ["6 benchmark reviews", "3 tricky cases", "One-click analyze"],
            compact=True,
        ),
        unsafe_allow_html=True,
    )

    _section(
        "Clear positive and negative reviews",
        "Each card is loaded into the analyzer automatically when you press the button.",
    )
    columns = st.columns(2, gap="large")
    for index, sample in enumerate(SAMPLES):
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(
                    f'<span class="sample-tag {sample["label"]}">{sample["label"]} example</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="sample-quote">“{_esc(sample["text"])}”</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="sample-note">Reference label: {_esc(sample["label"].title())} · {_esc(sample["title"])}</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Analyze this review →",
                    key=f"sample_{index}",
                    width="stretch",
                ):
                    _load_sample(sample["text"])

    _section(
        "Tricky examples",
        "These are useful for discussing limitations: mixed opinions, sarcasm and text where the model should sound less certain.",
    )
    tricky_cols = st.columns(3, gap="medium")
    for index, sample in enumerate(TRICKY_SAMPLES):
        with tricky_cols[index]:
            with st.container(border=True):
                st.markdown(
                    '<span class="sample-tag tricky">challenge</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="sample-quote">“{_esc(sample["text"])}”</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="sample-note">{_esc(sample["note"])}</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Test this example →",
                    key=f"tricky_{index}",
                    width="stretch",
                ):
                    _load_sample(sample["text"])

    st.markdown(
        """
<div class="notice">
    <strong>Teaching point:</strong> the model is a binary classifier trained on movie reviews.
    It has no neutral class and cannot reliably understand sarcasm or subtle mixed opinions.
    A confident-looking probability is not the same thing as true language understanding.
    The confidence score lets you see when the model is uncertain.
</div>
""",
        unsafe_allow_html=True,
    )
    _footer()


# ---------------------------------------------------------------------------
# Navigation (streamlit multi-page)
# ---------------------------------------------------------------------------
HOME_PAGE = st.Page(render_home, title="Home", url_path="home", default=True)
ANALYZER_PAGE = st.Page(render_analyzer, title="Sentiment Analyzer", url_path="analyzer")
PERFORMANCE_PAGE = st.Page(render_performance, title="Model Performance", url_path="performance")
SAMPLES_PAGE = st.Page(render_samples, title="Sample Reviews", url_path="samples")


def main() -> None:
    _inject_css()
    navigation = st.navigation(
        [HOME_PAGE, ANALYZER_PAGE, PERFORMANCE_PAGE, SAMPLES_PAGE],
        position="sidebar",
    )
    _sidebar_bottom()
    navigation.run()


if __name__ == "__main__":
    main()
