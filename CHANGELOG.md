# Changelog

All notable changes to this project are documented here.

## [1.1.0] - 2026-11-21
### Added
- Redesigned web interface: hero banner, coloured result card, spam probability with confidence label,
  side-by-side answers of all three models, word-level explanation, adjustable spam threshold, model selector.
- Clickable spam and genuine example messages.
- Model performance tab with metric cards, comparison table, ROC curves and confusion matrices.
- Unseen-message test: 60 hand-written messages (modern Indian scams, Hinglish, genuine OTP and bank alerts)
  that are not in the training data; results saved to `reports/unseen_results.csv` and shown in the app.
- `SpamClassifier.explain()` and `Prediction.confidence`.

## [1.0.0] - 2026-11-21
### Added
- Installable Python package `spamdetector` (src layout) with a `spam-detector` command-line tool.
- Three models (Multinomial Naive Bayes, Logistic Regression, calibrated linear SVM) on TF-IDF features.
- Hyper-parameter tuning with 5-fold stratified cross-validation; model chosen by cross-validated F1.
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC/PR curves,
  top spam words and an error-analysis file.
- Streamlit web interface with spam probability.
- Unit tests (pytest), linting (ruff) and GitHub Actions CI.
- Model card, project report and viva questions.
