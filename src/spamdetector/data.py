"""Loading, validating and splitting the dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from spamdetector.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)

_LABEL_TO_TARGET = {"ham": 0, "spam": 1}


def load_dataset(path: Path | str = DATA_PATH, drop_duplicates: bool = True) -> pd.DataFrame:
    """Read the SMS dataset and return a clean DataFrame.

    The returned frame has the columns ``label`` ("ham"/"spam"), ``message`` (text)
    and ``target`` (0 = ham, 1 = spam). Both the Kaggle layout (``v1``, ``v2``) and a
    simple ``label``/``message`` layout are accepted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the CSV there or run scripts/download_data.py."
        )

    try:
        raw = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        raw = pd.read_csv(path, encoding="latin-1")

    if {"v1", "v2"}.issubset(raw.columns):
        raw = raw.rename(columns={"v1": "label", "v2": "message"})
    if not {"label", "message"}.issubset(raw.columns):
        raise ValueError("The CSV must contain the columns 'label' and 'message' (or 'v1' and 'v2').")

    data = raw[["label", "message"]].dropna().copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()

    unknown = set(data["label"]) - set(_LABEL_TO_TARGET)
    if unknown:
        raise ValueError(f"Unexpected labels found: {sorted(unknown)}. Expected only 'ham' and 'spam'.")

    if drop_duplicates:
        before = len(data)
        data = data.drop_duplicates(subset="message")
        logger.info("Removed %d duplicate messages", before - len(data))

    data["target"] = data["label"].map(_LABEL_TO_TARGET)
    return data.reset_index(drop=True)


def split_dataset(data: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    return train_test_split(
        data["message"],
        data["target"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["target"],
    )
