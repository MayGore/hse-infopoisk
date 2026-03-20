import numpy as np
import pandas as pd

from indexing.base import (
    preprocess_query,
    tokenize_pp_series,
    top_k_from_scores,
    validate_df_columns,
)


def empty_result() -> pd.DataFrame:
    """Return empty search result with standard columns."""
    return pd.DataFrame(columns=["doc_id", "score", "text"])


def vectorize_tokens(
    tokens,
    get_token_vector,  # specific to type of indexing
    vector_size: int,
    dtype=np.float32,
):
    """
    Convert token list into one dense vector.

    Strategy:
    - get vectors for all tokens found in the model
    - average them
    - return None if no token vectors are available
    """
    vectors = []

    for token in tokens:
        vec = get_token_vector(token)
        if vec is None:
            continue

        vec = np.asarray(vec, dtype=dtype)
        if vec.shape != (vector_size,):
            continue

        vectors.append(vec)

    if not vectors:
        return None

    matrix = np.vstack(vectors)
    return matrix.mean(axis=0).astype(dtype)


def build_doc_vectors(
    df: pd.DataFrame,
    get_token_vector,
    vector_size: int,
    text_pp_col: str = "text_pp",
    dtype=np.float32,
):
    """
    Build dense document matrix from preprocessed texts.

    Returns:
    - doc_vectors: np.ndarray, shape (n_docs, vector_size)
    - doc_norms: np.ndarray, shape (n_docs,)
    """
    validate_df_columns(df, [text_pp_col])

    docs = tokenize_pp_series(df[text_pp_col].fillna("").astype(str).tolist())
    doc_vectors = np.zeros((len(docs), vector_size), dtype=dtype)

    for doc_id, tokens in enumerate(docs):
        vec = vectorize_tokens(
            tokens=tokens,
            get_token_vector=get_token_vector,
            vector_size=vector_size,
            dtype=dtype,
        )
        if vec is None:
            continue
        doc_vectors[doc_id] = vec

    doc_norms = np.linalg.norm(doc_vectors, axis=1)
    return doc_vectors, doc_norms


def search_semantic(
    query: str,
    texts,
    doc_vectors: np.ndarray,
    doc_norms: np.ndarray,
    get_token_vector,
    vector_size: int,
    top_k: int = 10,
    use_preprocessing: bool = True,
    dtype=np.float32,
) -> pd.DataFrame:
    """
    Search top documents using cosine similarity.

    Returns a DataFrame with columns: doc_id, score, text.
    """
    if not query or not str(query).strip():
        return empty_result()

    if top_k <= 0:
        return empty_result()

    if doc_vectors.shape[0] == 0:
        return empty_result()

    q_terms = preprocess_query(query, use_preprocessing=use_preprocessing)
    if not q_terms:
        return empty_result()

    q_vec = vectorize_tokens(
        tokens=q_terms,
        get_token_vector=get_token_vector,
        vector_size=vector_size,
        dtype=dtype,
    )
    if q_vec is None:
        return empty_result()

    q_norm = float(np.linalg.norm(q_vec))
    if q_norm == 0.0:
        return empty_result()

    dots = doc_vectors @ q_vec
    denom = doc_norms * q_norm

    scores = np.divide(
        dots,
        denom,
        out=np.zeros_like(dots, dtype=dtype),
        where=denom > 0,
    )

    top = top_k_from_scores(scores, top_k=top_k)
    if not top:
        return empty_result()

    rows = [{"doc_id": i, "score": s, "text": texts[i]} for i, s in top]
    return pd.DataFrame(rows)