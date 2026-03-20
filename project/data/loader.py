from pathlib import Path
import pandas as pd
import re

# Local paths (next to this file)
DATA_DIR = Path(__file__).resolve().parent
FULL_PATH = DATA_DIR / "data.csv"
SHORT_PATH = DATA_DIR / "data_short.csv"

# Remote dataset (HF)
HF_URI = (
    "hf://datasets/ScoutieAutoML/recipes_for_dishes_and_food_with_vectors_sentiment_ners/"
    "scoutieDataset_russian_receipt_with_vectors.csv"
)

TEXT_COL = "text"


def _clean_text(s: str) -> str:
    """Remove line breaks and normalize whitespace."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ")
    s = s.replace("\ufeff", " ").replace("\u200b", " ")  # BOM / zero-width
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _validate_text_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 'text', drop NaNs, cast to str, basic cleaning for valid csv."""
    if TEXT_COL not in df.columns:
        raise ValueError(f"В датасете нет колонки '{TEXT_COL}'")

    df = df[[TEXT_COL]].dropna().copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).map(_clean_text)
    df = df.reset_index(drop=True)
    return df


def load_full(force_download: bool = False) -> pd.DataFrame:
    """
    Load full dataset.
    If local data.csv exists (and force_download is False), read it.
    Otherwise download from HF and save to data.csv.
    """
    if FULL_PATH.exists() and not force_download:
        df = pd.read_csv(FULL_PATH, usecols=[TEXT_COL])
        return _validate_text_df(df)

    # Download only 'text' to speed up loading
    try:
        df = pd.read_csv(HF_URI, usecols=[TEXT_COL])
    except Exception as e:
        raise RuntimeError(
            "Не удалось скачать датасет с HuggingFace. "
            "Проверь интернет и установлены ли зависимости для чтения hf://"
        ) from e

    df = _validate_text_df(df)
    df.to_csv(FULL_PATH, index=False)
    return df


def form_short(sample_size: int = 2000, random_state: int = 42, min_len: int = 150) -> pd.DataFrame:
    """
    Create a smaller dataset (random sample) with texts at least `min_len` chars long
    and save to data_short.csv.
    """
    df_full = load_full(force_download=False)

    # Filter by text length
    lens = df_full[TEXT_COL].astype(str).str.len()
    df_full = df_full[lens >= min_len].reset_index(drop=True)

    if df_full.empty:
        raise ValueError(f"После фильтрации не осталось текстов длиной >= {min_len} символов")

    n = min(sample_size, len(df_full))
    df_short = df_full.sample(n=n, random_state=random_state).reset_index(drop=True)

    df_short.to_csv(SHORT_PATH, index=False, encoding="utf-8")
    return df_short


def load_short(auto_create: bool = True) -> pd.DataFrame:
    """
    Load data_short.csv. If it doesn't exist and auto_create is True, create it.
    """
    if not SHORT_PATH.exists():
        if not auto_create:
            raise FileNotFoundError("Файл data_short.csv не найден")
        return form_short()

    df = pd.read_csv(SHORT_PATH, usecols=[TEXT_COL])
    return _validate_text_df(df)

# testing

# df = form_short()
# print(df.head())
