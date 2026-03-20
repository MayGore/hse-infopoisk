from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

from indexing.base import (
    build_tf_csr,
    build_vocab,
    preprocess_query,
    top_k_from_scores,
    tokenize_pp_series,
    validate_df_columns,
)


class BM25MatrixIndex:
    """
    BM25 inverted index via sparse matrices (matrix implementation).

    - Documents are taken from df[text_pp_col], where text is already preprocessed and stored as
      a space-joined string: "lemma1 lemma2 ...".
    - TF matrix: raw counts, shape (n_docs, n_terms)
    - IDF (seminar-style): idf = 1 + log((N + 1) / (df + 1))
    - Scoring:
        score(doc, q) = sum_{t in q} idf(t) * (tf * (k1 + 1)) /
                        (tf + k1 * (1 - b + b * dl/avgdl))
      Query term frequency (qtf) is intentionally NOT used.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_pp_col: str = "text_pp",
        text_col: str = "text",
        k1: float = 1.5,
        b: float = 0.75,
        dtype: np.dtype = np.float32,
    ) -> None:
        validate_df_columns(df, [text_pp_col, text_col])

        if k1 <= 0:
            raise ValueError("k1 должен быть > 0")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b должен быть в диапазоне [0, 1]")

        self.text_pp_col = text_pp_col
        self.text_col = text_col
        self.k1 = float(k1)
        self.b = float(b)
        self.dtype = dtype

        # Store original texts for readable output
        self._texts: List[str] = df[text_col].astype(str).tolist()

        # Tokenize preprocessed documents
        docs = tokenize_pp_series(df[text_pp_col].fillna("").astype(str).tolist())
        self._doc_lens = np.asarray([len(d) for d in docs], dtype=np.float32)

        self._n_docs = int(len(docs))
        self._avgdl = float(self._doc_lens.mean()) if self._n_docs > 0 else 1.0
        if self._avgdl <= 0:
            self._avgdl = 1.0

        # Build vocabulary and TF matrix
        self._vocab, self._term2id = build_vocab(docs)
        self._tf_csr: csr_matrix = build_tf_csr(docs, self._term2id, dtype=self.dtype)

        # For efficient per-term access (columns)
        self._tf_csc: csc_matrix = self._tf_csr.tocsc()

        # Precompute document frequency df(t) for each term
        # df is the number of documents with tf > 0 for the term.
        df_vec = np.diff(self._tf_csc.indptr).astype(np.float32)

        # Precompute seminar-style IDF: 1 + log((N + 1)/(df + 1))
        n = float(self._n_docs)
        self._idf = (1.0 + np.log((n + 1.0) / (df_vec + 1.0))).astype(np.float32)

    @property
    def vocab(self) -> List[str]:
        """Sorted list of all indexed terms."""
        return self._vocab

    @property
    def doc_lens(self) -> List[int]:
        """Document lengths in tokens/lemmas."""
        return self._doc_lens.astype(int).tolist()

    @property
    def avgdl(self) -> float:
        """Average document length."""
        return float(self._avgdl)

    @property
    def tf_matrix(self) -> csr_matrix:
        """Document-term TF matrix (CSR)."""
        return self._tf_csr

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_preprocessing: bool = True,
    ) -> pd.DataFrame:
        """
        Search top documents for a query using BM25.

        Returns a DataFrame with columns: doc_id, score, text.
        """
        q_terms = preprocess_query(query, use_preprocessing=use_preprocessing)
        if not q_terms:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        n_terms = len(self._vocab)
        if n_terms == 0 or self._n_docs == 0:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        # BM25 usually treats query as a set; ignore duplicates
        q_terms = list(dict.fromkeys(q_terms))

        # Map query terms to term ids
        term_ids = [self._term2id[t] for t in q_terms if t in self._term2id]
        if not term_ids:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        scores = np.zeros(self._n_docs, dtype=np.float32)

        k1 = self.k1
        b = self.b
        avgdl = float(self._avgdl)
        doc_lens = self._doc_lens

        # Add contributions term-by-term using CSC columns (fast access)
        for term_id in term_ids:
            col_start = self._tf_csc.indptr[term_id]
            col_end = self._tf_csc.indptr[term_id + 1]
            if col_start == col_end:
                continue

            doc_ids = self._tf_csc.indices[col_start:col_end]
            tf_vals = self._tf_csc.data[col_start:col_end].astype(np.float32, copy=False)

            idf = float(self._idf[term_id])
            if idf == 0.0:
                continue

            dl = doc_lens[doc_ids]
            denom = tf_vals + k1 * (1.0 - b + b * (dl / avgdl))

            contrib = idf * (tf_vals * (k1 + 1.0)) / denom
            scores[doc_ids] += contrib.astype(np.float32, copy=False)

        top = top_k_from_scores(scores, top_k=top_k)
        if not top:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        rows = [{"doc_id": i, "score": s, "text": self._texts[i]} for i, s in top]
        return pd.DataFrame(rows)
