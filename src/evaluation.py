"""Model evaluation and chart generation.

All numbers shown in the app and report are produced here from the untouched
test split. Nothing is hard-coded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files without needing a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import ID_TO_LABEL

LABEL_NAMES = ["Negative", "Positive"]
COLORS = {"negative": "#DC2626", "positive": "#16A34A"}
MODEL_DISPLAY_NAMES = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def compute_metrics(y_true, y_pred) -> dict:
    """Calculate all metrics used by the project.

    Top-level precision/recall/F1 are weighted averages across the two
    classes. Positive-class metrics and a full per-class report are included
    as well, so nothing is hidden behind a single number.
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": _round(accuracy_score(y_true, y_pred)),
        "precision": _round(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": _round(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": _round(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": _round(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": _round(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": _round(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_class": {
            "precision": _round(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": _round(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1_score": _round(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        },
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": report,
        "support": int(len(y_true)),
    }


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart showing the number of positive and negative reviews."""

    counts = df["sentiment"].value_counts().reindex(["negative", "positive"])
    total = int(counts.sum())
    percentages = (counts / total * 100).round(1)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(
        ["Negative", "Positive"],
        counts.values,
        color=[COLORS["negative"], COLORS["positive"]],
        width=0.55,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.012,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#111827",
        )

    ax.set_title("Class Distribution of the Cleaned Dataset")
    ax.set_ylabel("Number of reviews")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_xlabel("")
    sns.despine(ax=ax)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: dict[str, dict], output_path: Path) -> None:
    """Grouped bar chart comparing both models on the four main metrics."""

    metric_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-score",
    }
    rows = []
    for model_key, metrics in results.items():
        for metric_key, metric_label in metric_names.items():
            rows.append(
                {
                    "Model": MODEL_DISPLAY_NAMES.get(model_key, model_key),
                    "Metric": metric_label,
                    "Score": metrics[metric_key],
                }
            )
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#6366F1", "#14B8A6"],
        ax=ax,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.set_ylim(0, 1.08)
    ax.set_title("Naive Bayes vs Logistic Regression (Test Set)")
    ax.set_xlabel("")
    ax.set_ylabel("Score (higher is better)")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)

    ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=2,
        frameon=False,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, model_name: str, output_path: Path) -> None:
    """Heatmap of true vs predicted labels."""

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.9, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=2,
        linecolor="white",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def extract_top_features(
    model,
    vectorizer,
    model_key: str,
    output_path: Path,
    top_n: int = 15,
) -> dict | None:
    """Find the tokens that push the model most strongly each way.

    Logistic Regression exposes coefficients; Naive Bayes exposes per-class
    log-probabilities. The difference between the two classes is comparable
    in spirit (how strongly a feature favours "positive").
    """

    try:
        feature_names = np.asarray(vectorizer.get_feature_names_out())
        if model_key == "logistic_regression":
            weights = np.asarray(model.coef_[0])
            score_label = "Logistic Regression coefficient"
        elif model_key == "naive_bayes":
            weights = np.asarray(model.feature_log_prob_[1] - model.feature_log_prob_[0])
            score_label = "Log-probability difference"
        else:
            return None
    except Exception:
        return None

    pos_idx = np.argsort(weights)[-top_n:][::-1]
    neg_idx = np.argsort(weights)[:top_n]

    top_features = {
        "positive": [
            {"token": str(feature_names[i]), "weight": _round(weights[i], 4)}
            for i in pos_idx
        ],
        "negative": [
            {"token": str(feature_names[i]), "weight": _round(weights[i], 4)}
            for i in neg_idx
        ],
        "score_label": score_label,
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    palette = {"positive": COLORS["positive"], "negative": COLORS["negative"]}
    for ax, side, title in (
        (axes[0], "positive", f"Top {top_n} Positive Features"),
        (axes[1], "negative", f"Top {top_n} Negative Features"),
    ):
        tokens = [item["token"] for item in top_features[side]][::-1]
        values = [item["weight"] for item in top_features[side]][::-1]
        ax.barh(tokens, values, color=palette[side], alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel(
            "Coefficient" if model_key == "logistic_regression" else "Log-prob. difference"
        )
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.25)
        sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return top_features


def save_classification_report(
    model_name: str,
    report: dict,
    output_path: Path,
) -> None:
    """Write a readable per-class report to a text file."""

    df = pd.DataFrame(report).transpose()
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Classification Report - {model_name}\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(df.to_string(float_format=lambda value: f"{value:.4f}"))
        handle.write("\n\nNotes:\n")
        handle.write("- precision/recall/f1 columns per class are shown above.\n")
        handle.write("- macro avg gives every class equal weight.\n")
        handle.write("- weighted avg accounts for class support (size).\n")
