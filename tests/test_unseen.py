import pandas as pd
import pytest

from spamdetector.config import DATA_PATH, UNSEEN_PATH
from spamdetector.data import load_dataset
from spamdetector.models import get_candidates
from spamdetector.preprocessing import clean_text
from spamdetector.unseen import evaluate_on_unseen, load_unseen


def test_unseen_file_is_balanced_and_valid():
    data = load_unseen(UNSEEN_PATH)
    assert len(data) >= 50
    assert set(data["label"]) == {"ham", "spam"}
    assert data["label"].value_counts().min() >= 25
    assert data["message"].is_unique


def test_unseen_messages_do_not_appear_in_training_data():
    """The whole point of the set: none of these messages may be in the dataset."""
    training = {clean_text(m) for m in load_dataset(DATA_PATH)["message"]}
    unseen = {clean_text(m) for m in load_unseen(UNSEEN_PATH)["message"]}
    assert not training & unseen


def test_overlapping_messages_are_dropped(tmp_path):
    path = tmp_path / "u.csv"
    pd.DataFrame({"label": ["spam", "ham"], "category": ["a", "b"],
                  "message": ["Free prize now", "See you tomorrow"]}).to_csv(path, index=False)
    kept = load_unseen(path, training_messages=["free prize now!!"])
    assert list(kept["message"]) == ["See you tomorrow"]


def test_bad_label_rejected(tmp_path):
    path = tmp_path / "u.csv"
    pd.DataFrame({"label": ["maybe"], "category": ["a"], "message": ["hi"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_unseen(path)


def test_evaluate_on_unseen_returns_metrics_and_table(tiny_dataframe):
    pipeline, _ = get_candidates(quick=True)["Naive Bayes"]
    pipeline.set_params(tfidf__min_df=1)
    tiny = tiny_dataframe.assign(target=lambda d: (d["label"] == "spam").astype(int))
    pipeline.fit(tiny["message"], tiny["target"])
    unseen = load_unseen(UNSEEN_PATH)
    metrics, table = evaluate_on_unseen({"Naive Bayes": pipeline}, unseen)
    assert 0.0 <= metrics["Naive Bayes"]["accuracy"] <= 1.0
    assert "Naive Bayes spam probability" in table.columns
    assert len(table) == len(unseen)
