import pandas as pd
import pytest

from spamdetector.data import load_dataset, split_dataset


def test_load_dataset_adds_target_and_drops_duplicates(tiny_csv):
    data = load_dataset(tiny_csv)
    assert set(data["target"]) == {0, 1}
    assert data["message"].is_unique
    assert len(data) == 12          # 6 spam + 6 ham unique messages


def test_load_accepts_kaggle_column_names(tmp_path):
    path = tmp_path / "kaggle.csv"
    pd.DataFrame({"v1": ["ham", "spam"], "v2": ["hello there", "free prize now"]}).to_csv(path, index=False)
    assert list(load_dataset(path)["label"]) == ["ham", "spam"]


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")


def test_unknown_label_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"label": ["maybe"], "message": ["hello"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_split_is_stratified(tiny_csv):
    data = load_dataset(tiny_csv)
    _, _, y_train, y_test = split_dataset(data, test_size=0.34)
    assert y_train.sum() > 0 and y_test.sum() > 0
