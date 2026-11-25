"""Train, tune, compare and save the spam models."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from spamdetector import config
from spamdetector.data import load_dataset, split_dataset
from spamdetector.evaluate import (
    compute_metrics,
    plot_class_distribution,
    plot_confusion_matrices,
    plot_message_length,
    plot_model_comparison,
    plot_roc_pr,
    plot_top_words,
)
from spamdetector.models import get_candidates
from spamdetector.unseen import evaluate_on_unseen, load_unseen

logger = logging.getLogger(__name__)


def train(
    quick: bool = False,
    data_path: Path | str = config.DATA_PATH,
    output_root: Path | str | None = None,
    unseen_path: Path | str = config.UNSEEN_PATH,
) -> dict:
    """Run the full experiment and return the results dictionary.

    1. load and clean the data, 2. split 80/20 (stratified),
    3. tune each model with 5-fold cross-validation on the training part only,
    4. evaluate once on the untouched test part,
    5. save the model that has the best cross-validated F1-score, plus reports and figures.

    The best model is chosen with cross-validation scores (not test scores) so the test set
    stays an honest, unbiased final check.
    """
    root = Path(output_root) if output_root else config.ROOT_DIR
    models_dir, reports_dir = root / "models", root / "reports"
    figures_dir = reports_dir / "figures"
    for folder in (models_dir, figures_dir):
        folder.mkdir(parents=True, exist_ok=True)

    data = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(data)
    logger.info("Messages: %d (train %d, test %d)", len(data), len(X_train), len(X_test))

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    results, curves, fitted = {}, {}, {}

    for name, (pipeline, grid) in get_candidates(quick).items():
        logger.info("Tuning %s ...", name)
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=cv, n_jobs=-1)
        started = time.perf_counter()
        search.fit(X_train, y_train)
        seconds = time.perf_counter() - started

        best = search.best_estimator_
        probabilities = best.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results[name] = {
            "best_params": {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()
            },
            "cv_f1_mean": float(search.best_score_),
            "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "train_seconds": round(seconds, 1),
            "test": compute_metrics(y_test, predictions, probabilities),
        }
        curves[name] = (y_test.to_numpy(), probabilities)
        fitted[name] = best
        logger.info("%s: CV F1 %.4f | test F1 %.4f", name,
                    results[name]["cv_f1_mean"], results[name]["test"]["f1"])

    best_name = max(results, key=lambda n: results[n]["cv_f1_mean"])

    # ---------- reports ----------
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "quick_mode": quick,
        "dataset": {
            "messages_after_dedup": int(len(data)),
            "ham": int((data["target"] == 0).sum()),
            "spam": int((data["target"] == 1).sum()),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        "best_model": best_name,
        "selection_rule": "highest mean cross-validated F1 on the training set",
        "models": results,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    table = pd.DataFrame([
        {
            "Model": name,
            "CV F1": res["cv_f1_mean"],
            "Accuracy": res["test"]["accuracy"],
            "Precision": res["test"]["precision"],
            "Recall": res["test"]["recall"],
            "F1": res["test"]["f1"],
            "ROC AUC": res["test"]["roc_auc"],
            "PR AUC": res["test"]["pr_auc"],
        }
        for name, res in results.items()
    ])
    table.round(4).to_csv(reports_dir / "model_comparison.csv", index=False)

    # Error analysis: which test messages does the chosen model get wrong?
    best_prob = fitted[best_name].predict_proba(X_test)[:, 1]
    errors = pd.DataFrame({"message": X_test.to_numpy(), "actual": y_test.map(config.LABEL_NAMES).to_numpy(),
                           "spam_probability": best_prob.round(4)})
    errors = errors[(errors["spam_probability"] >= 0.5) != (errors["actual"] == "spam")]
    errors.sort_values("spam_probability").to_csv(reports_dir / "misclassified_examples.csv", index=False)

    # ---------- extra check: messages the models have never seen ----------
    if Path(unseen_path).exists():
        unseen = load_unseen(unseen_path, training_messages=data["message"])
        unseen_metrics, unseen_table = evaluate_on_unseen(fitted, unseen)
        summary["unseen"] = {"messages": int(len(unseen)), "models": unseen_metrics}
        unseen_table.to_csv(reports_dir / "unseen_results.csv", index=False)
        (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    # ---------- figures ----------
    plot_class_distribution(data, figures_dir / "class_distribution.png")
    plot_message_length(data, figures_dir / "message_length.png")
    plot_model_comparison(results, figures_dir / "model_comparison.png")
    plot_confusion_matrices(results, figures_dir / "confusion_matrices.png")
    plot_roc_pr(curves, figures_dir / "roc_curves.png", figures_dir / "pr_curves.png")

    lr = fitted["Logistic Regression"]
    words = lr.named_steps["tfidf"].get_feature_names_out()
    coefficients = lr.named_steps["clf"].coef_[0]
    top = coefficients.argsort()[::-1][:15]
    plot_top_words([words[i] for i in top], [float(coefficients[i]) for i in top],
                   figures_dir / "top_spam_words.png", "Top 15 words that indicate spam")

    # ---------- save the chosen model ----------
    joblib.dump(fitted[best_name], models_dir / "spam_model.joblib")
    joblib.dump(fitted, models_dir / "all_models.joblib")   # lets the web app compare all three models
    metadata = {k: summary[k] for k in ("created_utc", "sklearn_version", "best_model")}
    metadata["best_params"] = results[best_name]["best_params"]
    metadata["test_metrics"] = results[best_name]["test"]
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info("Best model: %s. Saved to %s", best_name, models_dir / "spam_model.joblib")
    return summary
