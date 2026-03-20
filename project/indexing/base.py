from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from utils.preprocessing import preprocess_text


def validate_df_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    """Validate that dataframe contains all required columns."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"В датафрейме не хватает колонок: {missing}")


def tokenize_pp_series(texts_pp: Iterable[str]) -> List[List[str]]:
    """Split preprocessed texts (space-joined lemmas) into token lists."""
    return [str(s).split() if s is not None else [] for s in texts_pp]


def build_vocab(docs: Sequence[Sequence[str]]) -> Tuple[List[str], Dict[str, int]]:
    """
    Build sorted vocabulary and term->id mapping.

    Sorting is used for deterministic term ids across runs.
    """
    all_terms = set()
    for doc in docs:
        all_terms.update(doc)

    vocab = sorted(all_terms)
    term2id = {t: i for i, t in enumerate(vocab)}
    return vocab, term2id


def build_tf_csr(
    docs: Sequence[Sequence[str]],
    term2id: Dict[str, int],
    dtype: np.dtype = np.float32,
) -> csr_matrix:
    """
    Build a document-term sparse matrix with raw term frequencies (TF).

    Output shape: (n_docs, n_terms)
    """
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    n_docs = len(docs)
    n_terms = len(term2id)

    for doc_id, doc in enumerate(docs):
        tf = Counter(doc)
        for term, cnt in tf.items():
            term_id = term2id.get(term)
            if term_id is None:
                continue
            rows.append(doc_id)
            cols.append(term_id)
            data.append(float(cnt))

    return csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms), dtype=dtype)


def preprocess_query(query: str, use_preprocessing: bool = True) -> List[str]:
    """Preprocess query to match index vocabulary (lemmas)."""
    if not query or not str(query).strip():
        return []
    if use_preprocessing:
        return preprocess_text(query)
    return str(query).lower().split()


def build_query_vector(
    q_terms: Sequence[str],
    term2id: Dict[str, int],
    n_terms: int,
    dtype: np.dtype = np.float32,
) -> csr_matrix:
    """
    Build a sparse query vector with raw counts.

    Output shape: (n_terms, 1)
    """
    q_tf = Counter(q_terms)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for term, cnt in q_tf.items():
        term_id = term2id.get(term)
        if term_id is None:
            continue
        rows.append(term_id)
        cols.append(0)
        data.append(float(cnt))

    return csr_matrix((data, (rows, cols)), shape=(n_terms, 1), dtype=dtype)


def top_k_from_scores(scores: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    """
    Return (doc_id, score) pairs for top_k scores > 0.

    Sorting: score desc, doc_id asc (stable output).
    """
    if top_k <= 0:
        return []

    idx = np.flatnonzero(scores > 0)
    if idx.size == 0:
        return []

    vals = scores[idx]
    k = min(top_k, idx.size)

    # Partial selection + final sort for speed
    part = np.argpartition(-vals, kth=k - 1)[:k]
    cand_ids = idx[part]
    cand_scores = scores[cand_ids]

    order = np.lexsort((cand_ids, -cand_scores))
    cand_ids = cand_ids[order]
    cand_scores = cand_scores[order]

    return [(int(i), float(s)) for i, s in zip(cand_ids, cand_scores)]