"""Central place for paths and settings used across the project."""

from __future__ import annotations

import os
from pathlib import Path

# Project root = the folder that contains `src/`. It can be overridden with the
# SPAMDETECTOR_HOME environment variable (useful if the package is installed elsewhere).
ROOT_DIR = Path(os.environ.get("SPAMDETECTOR_HOME", Path(__file__).resolve().parents[2]))

DATA_PATH = ROOT_DIR / "data" / "raw" / "spam.csv"
UNSEEN_PATH = ROOT_DIR / "data" / "unseen" / "unseen_messages.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "spam_model.joblib"
ALL_MODELS_PATH = MODELS_DIR / "all_models.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
UNSEEN_RESULTS_PATH = REPORTS_DIR / "unseen_results.csv"

RANDOM_STATE = 42      # fixed seed so results are reproducible
TEST_SIZE = 0.2        # 20% of messages are kept aside for the final test
CV_FOLDS = 5           # number of cross-validation folds used for tuning
LABEL_NAMES = {0: "ham", 1: "spam"}
