"""Test the trained models on hand-written messages that were never part of the dataset.

The file ``data/unseen/unseen_messages.csv`` was written separately from the training data and is
never used for training or tuning. It deliberately contains modern Indian-style scams (KYC, UPI,
courier fees), Hinglish chat, and *legitimate* automatic messages (OTPs, bank alerts) that look
a bit like spam. That makes it a harder, more realistic check than the original test split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spamdetector.config import UNSEEN_PATH
from spamdetector.evaluate import compute_metrics
from spamdetector.preprocessing import clean_text


def load_unseen(path: Path | str = UNSEEN_PATH, training_messages=None) -> pd.DataFrame:
    """Read the unseen messages. Any message that also occurs in the training data is dropped."""
    data = pd.read_csv(path, encoding="utf-8")
    required = {"label", "category", "message"}
    if not required.issubset(data.columns):
        raise ValueError(f"The unseen-messages file must contain the columns {sorted(required)}.")
    data["target"] = data["label"].map({"ham": 0, "spam": 1})
    if data["target"].isna().any():
        raise ValueError("Unseen-messages labels must be 'ham' or 'spam'.")
    if training_messages is not None:
        seen = {clean_text(m) for m in training_messages}
        data = data[~data["message"].map(clean_text).isin(seen)]
    return data.reset_index(drop=True)


def evaluate_on_unseen(models: dict, unseen: pd.DataFrame, threshold: float = 0.5):
    """Score every model on the unseen messages.

    Returns ``(metrics, results)``: ``metrics`` maps model name -> metric dict (plus accuracy per
    category); ``results`` is a table with one row per message and one probability column per model.
    """
    results = unseen[["label", "category", "message"]].copy()
    metrics = {}
    for name, pipeline in models.items():
        probabilities = pipeline.predict_proba(unseen["message"])[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        results[f"{name} spam probability"] = probabilities.round(4)
        model_metrics = compute_metrics(unseen["target"], predictions, probabilities)
        correct = pd.Series(predictions == unseen["target"].to_numpy(), index=unseen.index)
        model_metrics["accuracy_by_category"] = {
            category: float(correct[unseen["category"] == category].mean())
            for category in sorted(unseen["category"].unique())
        }
        metrics[name] = model_metrics
    return metrics, results
