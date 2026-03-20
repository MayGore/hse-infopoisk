from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd
import pymorphy3

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


_morph = pymorphy3.MorphAnalyzer()
RU_STOPWORDS = stopwords.words("russian")
_TOKENIZER = RegexpTokenizer(r"[a-zа-яё]+", flags=re.IGNORECASE)  # removes emojis & punctuation


def clean_text(text: str) -> str:
    """Lowercase + normalize whitespace."""
    text = str(text).lower()
    text = text.replace("ё", "е")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    text = re.compile(r"\s+").sub(" ", text).strip()
    return text


def tokenize(text: str, min_token_len: int = 2) -> List[str]:
    """Tokenize using regex tokenizer and filter short tokens."""
    if not text:
        return []
    tokens = _TOKENIZER.tokenize(text)
    return [t for t in tokens if len(t) >= min_token_len]


@lru_cache(maxsize=200_000)
def lemmatize_token(token: str) -> str:
    """Lemmatize one token (cached for speed)."""
    return _morph.parse(token)[0].normal_form


def preprocess_text(
    text: str,
    min_token_len: int = 2,
) -> List[str]:
    """Full pipeline: clean -> tokenize -> lemmatize -> remove stopwords."""

    cleaned = clean_text(text)
    tokens = tokenize(cleaned, min_token_len=min_token_len)
    lemmas = [lemmatize_token(t) for t in tokens]
    lemmas = [w for w in lemmas if w not in RU_STOPWORDS]
    return lemmas


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    out_col: str = "text_pp",
) -> pd.DataFrame:
    """Add preprocessed text column (space-joined lemmas)."""
    if text_col not in df.columns:
        raise ValueError(f"В датафрейме нет колонки '{text_col}'")

    df = df.copy()
    df[out_col] = df[text_col].map(lambda x: " ".join(preprocess_text(x)))
    return df


def save_preprocessed(df_pp: pd.DataFrame, path: Path) -> None:
    """Save preprocessed dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df_pp.to_csv(path, index=False, encoding="utf-8")


def load_preprocessed(path: Path) -> pd.DataFrame:
    """Load preprocessed CSV and check required columns."""
    df = pd.read_csv(path)
    if "text_pp" not in df.columns:
        raise ValueError("В файле препроцессинга нет колонок 'text_pp' и/или 'doc_len'")
    return df


def get_or_create_preprocessed(df_raw: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Try to load preprocessed file, otherwise build and save it."""
    if out_path.exists():
        return load_preprocessed(out_path)

    df_pp = preprocess_dataframe(df_raw, text_col="text", out_col="text_pp")
    save_preprocessed(df_pp, out_path)
    return df_pp
