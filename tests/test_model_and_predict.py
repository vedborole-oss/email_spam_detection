import json

import joblib
import pytest

from spamdetector.data import load_dataset
from spamdetector.models import get_candidates
from spamdetector.predict import SpamClassifier
from spamdetector.train import train


@pytest.fixture()
def fitted_pipeline(tiny_csv):
    data = load_dataset(tiny_csv)
    pipeline, _ = get_candidates(quick=True)["Naive Bayes"]
    pipeline.set_params(tfidf__min_df=1)
    return pipeline.fit(data["message"], data["target"])


def test_classifier_separates_obvious_cases(fitted_pipeline):
    classifier = SpamClassifier(fitted_pipeline)
    assert classifier.predict("Free prize! Call now to claim your cash").is_spam
    assert not classifier.predict("Are you coming to the lecture tomorrow?").is_spam


def test_probability_is_between_0_and_1(fitted_pipeline):
    result = SpamClassifier(fitted_pipeline).predict("hello")
    assert 0.0 <= result.spam_probability <= 1.0


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_message_is_rejected(fitted_pipeline, bad):
    with pytest.raises(ValueError):
        SpamClassifier(fitted_pipeline).predict(bad)


def test_load_without_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="spam-detector train"):
        SpamClassifier.load(tmp_path / "missing.joblib")


def test_all_three_candidates_exist():
    assert set(get_candidates()) == {"Naive Bayes", "Logistic Regression", "SVM"}


def test_saved_model_round_trip(fitted_pipeline, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)
    assert SpamClassifier.load(path).predict("Free prize now").spam_probability > 0.5


def test_quick_training_end_to_end(tiny_csv, tmp_path):
    """Runs the whole pipeline on the tiny dataset and checks the output files exist."""
    summary = train(quick=True, data_path=tiny_csv, output_root=tmp_path)
    assert summary["best_model"] in summary["models"]
    assert (tmp_path / "models" / "spam_model.joblib").exists()
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert metrics["dataset"]["messages_after_dedup"] == 12
    assert (tmp_path / "reports" / "figures" / "model_comparison.png").exists()


def test_explain_returns_words_that_push_towards_spam(fitted_pipeline):
    words = dict(SpamClassifier(fitted_pipeline).explain("Free prize! Call now", top_n=5))
    assert words, "expected word-level explanations for Naive Bayes"
    assert any(score > 0 for score in words.values())


def test_confidence_label():
    from spamdetector.predict import Prediction

    assert Prediction("spam", True, 0.97).confidence_label == "High"
    assert Prediction("ham", False, 0.20).confidence_label == "Medium"
    assert Prediction("spam", True, 0.60).confidence_label == "Low"
