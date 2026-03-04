from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from indexing.base import (
    build_query_vector,
    build_tf_csr,
    build_vocab,
    preprocess_query,
    top_k_from_scores,
    tokenize_pp_series,
    validate_df_columns,
)


class FreqMatrixIndex:
    """
    Inverted index via document-term TF matrix (matrix implementation).

    - Documents are taken from df[text_pp_col], where text is already preprocessed and stored as
      a space-joined string: "lemma1 lemma2 ...".
    - Matrix: TF (raw counts), shape (n_docs, n_terms)
    - Search scoring: score = TF_matrix @ q_vector
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_pp_col: str = "text_pp",
        text_col: str = "text",
        dtype: np.dtype = np.float32,
    ) -> None:
        validate_df_columns(df, [text_pp_col, text_col])

        self.text_pp_col = text_pp_col
        self.text_col = text_col
        self.dtype = dtype

        # Store original texts for readable output
        self._texts: List[str] = df[text_col].astype(str).tolist()

        # Tokenize preprocessed documents
        docs = tokenize_pp_series(df[text_pp_col].fillna("").astype(str).tolist())
        self._doc_lens = [len(d) for d in docs]

        # Build vocabulary and TF matrix
        self._vocab, self._term2id = build_vocab(docs)
        self._tf: csr_matrix = build_tf_csr(docs, self._term2id, dtype=self.dtype)

    @property
    def vocab(self) -> List[str]:
        """Sorted list of all indexed terms."""
        return self._vocab

    @property
    def doc_lens(self) -> List[int]:
        """Document lengths in tokens/lemmas."""
        return self._doc_lens

    @property
    def matrix(self) -> csr_matrix:
        """Document-term TF matrix (CSR)."""
        return self._tf

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_preprocessing: bool = True,
    ) -> pd.DataFrame:
        """
        Search top documents for a query using TF matrix.

        Returns a DataFrame with columns: doc_id, score, text.
        """
        q_terms = preprocess_query(query, use_preprocessing=use_preprocessing)
        if not q_terms:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        n_terms = len(self._vocab)
        if n_terms == 0:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        q_vec = build_query_vector(q_terms, self._term2id, n_terms, dtype=self.dtype)
        if q_vec.nnz == 0:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        scores = (self._tf @ q_vec).toarray().ravel() # matrix mul
        top = top_k_from_scores(scores, top_k=top_k)

        rows = [{"doc_id": i, "score": s, "text": self._texts[i]} for i, s in top]
        return pd.DataFrame(rows)