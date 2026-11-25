"""spamdetector - classify SMS and email messages as spam or ham."""

from spamdetector.predict import Prediction, SpamClassifier

__version__ = "1.1.0"
__all__ = ["Prediction", "SpamClassifier", "__version__"]
