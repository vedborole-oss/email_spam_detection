"""Shared test fixtures: a tiny dataset so tests run in a second without the real data."""

import pandas as pd
import pytest

SPAM = [
    "WINNER! You have won a free prize. Call 09061701461 now to claim",
    "Free entry in a weekly competition. Text WIN to 80086 now",
    "URGENT! Your account is blocked. Click http://bit.ly/abc to verify",
    "Congratulations! You won 10000 cash. Claim your prize now",
    "Free ringtone! Reply STOP to unsubscribe. Txt to 87077",
    "You have been selected for a 500 gift voucher. Call now",
]
HAM = [
    "Hey, are you coming to college tomorrow?",
    "Please send me the lab record by evening",
    "I will call you after the lecture, ok?",
    "Can we meet at the canteen for lunch today",
    "Thanks for the notes, they were really helpful",
    "Don't forget to bring your laptop for the practical",
]


@pytest.fixture()
def tiny_dataframe():
    rows = [("spam", m) for m in SPAM] * 4 + [("ham", m) for m in HAM] * 4
    return pd.DataFrame(rows, columns=["label", "message"])


@pytest.fixture()
def tiny_csv(tmp_path, tiny_dataframe):
    path = tmp_path / "tiny.csv"
    tiny_dataframe.to_csv(path, index=False)
    return path
