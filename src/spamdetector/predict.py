"""Load a trained model and classify new messages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from spamdetector.config import ALL_MODELS_PATH, MODEL_PATH


@dataclass(frozen=True)
class Prediction:
    """Result for one message."""

    label: str              # "spam" or "ham"
    is_spam: bool
    spam_probability: float  # between 0 and 1

    @property
    def confidence(self) -> float:
        """How sure the model is about the label it chose (0.5 = a coin flip, 1.0 = certain)."""
        return max(self.spam_probability, 1.0 - self.spam_probability)

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.90:
            return "High"
        if self.confidence >= 0.75:
            return "Medium"
        return "Low"


class SpamClassifier:
    """Thin wrapper around the saved scikit-learn pipeline."""

    def __init__(self, pipeline, threshold: float = 0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    @classmethod
    def load(cls, path: Path | str | None = None, threshold: float = 0.5) -> "SpamClassifier":
        path = Path(path) if path else Path(MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Train one first with:  spam-detector train"
            )
        return cls(joblib.load(path), threshold=threshold)

    def predict(self, message: str) -> Prediction:
        return self.predict_many([message])[0]

    def predict_many(self, messages: list) -> list:
        if not messages or any(not isinstance(m, str) or not m.strip() for m in messages):
            raise ValueError("Messages must be non-empty text.")
        probabilities = self.pipeline.predict_proba(messages)[:, 1]
        return [
            Prediction(
                label="spam" if p >= self.threshold else "ham",
                is_spam=bool(p >= self.threshold),
                spam_probability=float(p),
            )
            for p in probabilities
        ]

    def explain(self, message: str, top_n: int = 5) -> list:
        """Words in the message that pushed the decision, as ``(word, score)`` pairs.

        A positive score pushes towards spam, a negative score towards ham. Works for Naive Bayes
        and Logistic Regression; returns an empty list for models that cannot be explained this way.
        """
        tfidf = self.pipeline.named_steps["tfidf"]
        classifier = self.pipeline.named_steps["clf"]
        if hasattr(classifier, "feature_log_prob_"):
            weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
        elif hasattr(classifier, "coef_"):
            weights = classifier.coef_[0]
        else:
            return []
        row = tfidf.transform([message]).tocoo()
        names = tfidf.get_feature_names_out()
        scores = [(str(names[j]), float(value * weights[j])) for j, value in zip(row.col, row.data)]
        return sorted(scores, key=lambda item: abs(item[1]), reverse=True)[:top_n]


def load_all_models(path: Path | str | None = None, threshold: float = 0.5) -> dict:
    """Load all three trained models as ``{name: SpamClassifier}`` (used by the web app)."""
    path = Path(path) if path else Path(ALL_MODELS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"No trained models at {path}. Train them first with:  spam-detector train")
    return {name: SpamClassifier(pipeline, threshold) for name, pipeline in joblib.load(path).items()}
