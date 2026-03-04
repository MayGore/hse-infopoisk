import math
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List

import pandas as pd

from utils.preprocessing import preprocess_text


class BM25DictIndex:
    """
    BM25 inverted index (dictionary implementation).

    - Documents are taken from df[text_pp_col], where text is already preprocessed and stored as
      a space-joined string: "lemma1 lemma2 ...".
    - Index structure: term -> {doc_id: tf}
    - Scoring:
        score(doc, q) = sum_{t in q} idf(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_pp_col: str = "text_pp",
        text_col: str = "text",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:

        if k1 <= 0:
            raise ValueError("k1 должен быть > 0")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b должен быть в диапазоне [0, 1]")

        self.text_pp_col = text_pp_col
        self.text_col = text_col
        self.k1 = float(k1)
        self.b = float(b)

        self._texts: List[str] = df[text_col].astype(str).tolist()

        # term -> {doc_id: tf}
        self._index: Dict[str, Dict[int, int]] = {}

        # term -> idf
        self._idf: Dict[str, float] = {}

        self._doc_lens: List[int] = []
        self._avgdl: float = 1.0
        self._vocab: List[str] = []

        self._build(df)

    @property
    def vocab(self) -> List[str]:
        """Sorted list of all indexed terms."""
        return self._vocab

    @property
    def doc_lens(self) -> List[int]:
        """Document lengths in tokens/lemmas."""
        return self._doc_lens

    @property
    def avgdl(self) -> float:
        """Average document length."""
        return self._avgdl

    def get_posting(self, term: str) -> Dict[int, int]:
        """Return posting list for a term: {doc_id: tf}."""
        return self._index.get(term, {})

    def get_idf(self, term: str) -> float:
        """Return IDF for a term (0.0 if term is unseen)."""
        return float(self._idf.get(term, 0.0))

    def _build(self, df: pd.DataFrame) -> None:
        # for each term keep tf per document
        postings: DefaultDict[str, Dict[int, int]] = defaultdict(dict)
        doc_lens: List[int] = []

        texts_pp = df[self.text_pp_col].fillna("").astype(str).tolist()

        for doc_id, text_pp in enumerate(texts_pp):
            tokens = text_pp.split()
            dl = len(tokens)
            doc_lens.append(dl)

            # Count term frequencies within the document
            tf = Counter(tokens)
            for term, cnt in tf.items():
                postings[term][doc_id] = int(cnt)

        self._index = dict(postings)
        self._vocab = sorted(self._index.keys())
        self._doc_lens = doc_lens

        n_docs = len(doc_lens)
        if n_docs == 0:
            self._avgdl = 1.0
            self._idf = {}
            return

        avgdl = sum(doc_lens) / n_docs
        # no division by zero if all documents are empty
        self._avgdl = float(avgdl) if avgdl > 0 else 1.0

        # Compute seminar-style IDF:
        #   idf = 1 + log((N + 1) / (df + 1))
        idf: Dict[str, float] = {}
        for term, posting in self._index.items():
            df_t = len(posting)  # document frequency
            val = 1.0 + math.log((n_docs + 1.0) / (df_t + 1.0))
            idf[term] = float(val)

        self._idf = idf

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_preprocessing: bool = True,
    ) -> pd.DataFrame:
        """
        Search top documents for a query using BM25.

        Returns a DataFrame with columns: doc_id, score, text.

        If use_preprocessing=True:
          query is passed through preprocess_text to match the index vocabulary.
        """
        if not query or not str(query).strip():
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        if top_k <= 0:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        if use_preprocessing:
            q_terms = preprocess_text(query)
        else:
            q_terms = str(query).lower().split()

        q_terms = list(dict.fromkeys(q_terms))
        if not q_terms:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        scores: DefaultDict[int, float] = defaultdict(float)

        for term in q_terms:
            posting = self._index.get(term)
            if not posting:
                continue

            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue

            for doc_id, tf in posting.items():
                dl = self._doc_lens[doc_id]

                # BM25 term score
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl))
                if denom == 0:
                    continue

                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / denom

        if not scores:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        # sort by score desc, then by doc_id asc for stable output
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        rows = [
            {"doc_id": doc_id, "score": score, "text": self._texts[doc_id]}
            for doc_id, score in top
        ]
        return pd.DataFrame(rows)
