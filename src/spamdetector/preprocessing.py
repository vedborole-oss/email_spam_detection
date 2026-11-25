"""Text cleaning. The same function is used for training and for prediction."""

from __future__ import annotations

import re

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_CURRENCY = re.compile(r"[£$€₹]")
_NUMBER = re.compile(r"\d+")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Return a simplified version of a message.

    Steps: lowercase -> links become ``urltoken`` -> currency signs become
    ``currencytoken`` -> numbers become ``numtoken`` -> everything that is not a
    letter is removed -> extra spaces are collapsed.

    Links, money and numbers are replaced by tokens (not deleted) because *having*
    a link, an amount or a phone number is itself a strong spam signal.
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = _URL.sub(" urltoken ", text)
    text = _CURRENCY.sub(" currencytoken ", text)
    text = _NUMBER.sub(" numtoken ", text)
    text = _NON_LETTER.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
