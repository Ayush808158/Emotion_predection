"""Step 1 - Data cleaning: load data/train.txt and clean the text.

Used by train_model.py (training) and app.py (so user input is cleaned the same way).
Run `python data_cleaning.py` to preview the cleaning on a few rows.
"""
import re
from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "train.txt"

_URL = re.compile(r"https?://\S+|www\.\S+")
_APOSTROPHES = re.compile(r"['\u2019`]")
_NON_LETTERS = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, remove URLs/apostrophes/punctuation/digits, collapse spaces.

    Stop words are deliberately KEPT: "not", "no", "very" change the emotion
    ("do not feel reassured" is not the same as "feel reassured").
    """
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _APOSTROPHES.sub("", text)      # didn't -> didnt (same style as the dataset)
    text = _NON_LETTERS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """Read 'text;emotion' lines and return a cleaned DataFrame."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Put train.txt inside the data/ folder.")
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ";" not in line:
            continue
        text, label = line.strip().rsplit(";", 1)      # split on the LAST ';'
        rows.append((text, label.strip().lower()))
    df = pd.DataFrame(rows, columns=["text", "emotion"])
    df["text"] = df["text"].map(clean_text)
    return df[df["text"] != ""].reset_index(drop=True)


if __name__ == "__main__":
    data = load_data()
    print(f"{len(data):,} rows\n")
    print(data["emotion"].value_counts().to_string(), "\n")
    print(data.head(5).to_string())
