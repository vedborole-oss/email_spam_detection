"""Download the SMS Spam Collection from the UCI repository and save it as data/raw/spam.csv.

Usage:  python scripts/download_data.py
The repository already contains a copy of the data, so this is only needed to refresh it.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "spam.csv"


def parse_uci_text(text: str) -> pd.DataFrame:
    """The UCI file has one message per line: ``<label><TAB><message>``."""
    rows = []
    for line in text.splitlines():
        if "\t" in line:
            label, message = line.split("\t", 1)
            rows.append((label.strip(), message.strip()))
    return pd.DataFrame(rows, columns=["label", "message"])


def main() -> None:
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as response:  # noqa: S310 (fixed https URL)
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    text = archive.read("SMSSpamCollection").decode("utf-8")
    data = parse_uci_text(text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    print(f"Saved {len(data)} messages to {OUTPUT}")


if __name__ == "__main__":
    main()
