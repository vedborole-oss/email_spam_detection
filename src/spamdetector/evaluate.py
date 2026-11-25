"""Metrics and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Standard classification metrics for the spam class (spam = 1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_distribution(data, path: Path) -> None:
    counts = data["label"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#C44E52"])
    for i, value in enumerate(counts.values):
        ax.text(i, value + 40, str(value), ha="center")
    ax.set(title="Ham vs Spam messages", xlabel="Class", ylabel="Number of messages")
    _save(fig, path)


def plot_message_length(data, path: Path) -> None:
    lengths = data["message"].str.len()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in (("ham", "#4C72B0"), ("spam", "#C44E52")):
        ax.hist(lengths[data["label"] == label], bins=40, alpha=0.6, label=label, color=color)
    ax.set(title="Message length: ham vs spam", xlabel="Characters", ylabel="Messages")
    ax.legend()
    _save(fig, path)


def plot_model_comparison(results: dict, path: Path) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1"]
    names = list(results)
    width = 0.2
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metric_names):
        values = [results[n]["test"][metric] for n in names]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=metric.capitalize())
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, names)
    ax.set(ylim=(0.7, 1.05), ylabel="Score", title="Model comparison on the test set")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    _save(fig, path)


def plot_confusion_matrices(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for ax, (name, res) in zip(np.atleast_1d(axes), results.items()):
        t = res["test"]
        matrix = [[t["tn"], t["fp"]], [t["fn"], t["tp"]]]
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    _save(fig, path)


def plot_roc_pr(curves: dict, roc_path: Path, pr_path: Path) -> None:
    """``curves`` maps model name -> (y_true, y_score)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, y_score):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    _save(fig, roc_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, label=f"{name} (AP = {average_precision_score(y_true, y_score):.3f})")
    ax.set(title="Precision-Recall curves", xlabel="Recall", ylabel="Precision", ylim=(0.5, 1.02))
    ax.legend(loc="lower left")
    _save(fig, pr_path)


def plot_top_words(words: list, weights: list, path: Path, title: str) -> None:
    """Horizontal bar chart of the most influential words."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(words[::-1], weights[::-1], color="#C44E52")
    ax.set(title=title, xlabel="Logistic regression coefficient")
    _save(fig, path)
