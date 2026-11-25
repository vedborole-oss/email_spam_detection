"""The candidate models and their hyper-parameter grids.

Every candidate is a scikit-learn Pipeline: TF-IDF -> classifier. Keeping the text
cleaning inside the TF-IDF step means a saved model can be given raw text directly.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spamdetector.preprocessing import clean_text


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(preprocessor=clean_text, stop_words="english")


def get_candidates(quick: bool = False) -> dict:
    """Return ``{name: (pipeline, param_grid)}`` for the three algorithms.

    ``quick=True`` shrinks every grid to a single setting. It is meant for fast
    smoke tests (for example in continuous integration), not for real results.
    """
    if quick:
        tfidf_grid = {"tfidf__ngram_range": [(1, 2)], "tfidf__min_df": [2]}
        nb_grid, lr_grid, svm_grid = {"clf__alpha": [0.1]}, {"clf__C": [10]}, {"clf__estimator__C": [1]}
    else:
        tfidf_grid = {"tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [1, 2]}
        nb_grid = {"clf__alpha": [0.05, 0.1, 0.5, 1.0]}
        lr_grid = {"clf__C": [1, 10, 100]}
        svm_grid = {"clf__estimator__C": [0.1, 1, 10]}

    return {
        "Naive Bayes": (
            Pipeline([("tfidf", _vectorizer()), ("clf", MultinomialNB())]),
            {**tfidf_grid, **nb_grid},
        ),
        "Logistic Regression": (
            Pipeline([
                ("tfidf", _vectorizer()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]),
            {**tfidf_grid, **lr_grid},
        ),
        # LinearSVC has no probabilities, so it is wrapped in a calibrator that
        # turns its scores into real spam probabilities.
        "SVM": (
            Pipeline([
                ("tfidf", _vectorizer()),
                ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
            ]),
            {**tfidf_grid, **svm_grid},
        ),
    }
