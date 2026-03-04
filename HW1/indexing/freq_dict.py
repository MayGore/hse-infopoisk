from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List

import pandas as pd

from utils.preprocessing import preprocess_text


class FreqDictIndex:
    """
    Inverted index based on raw term frequencies (dictionary implementation).

    - Documents are taken from df[text_pp_col], where text is already preprocessed and stored as
      a space-joined string: "lemma1 lemma2 ...".
    - Index structure: term -> {doc_id: tf}
    - Search scoring: score(doc) = sum_{term in query} tf(term, doc) * qtf(term)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_pp_col: str = "text_pp",
        text_col: str = "text",
    ) -> None:

        self.text_pp_col = text_pp_col
        self.text_col = text_col

        self._texts: List[str] = df[text_col].astype(str).tolist()

        # term -> {doc_id: tf}
        self._index: Dict[str, Dict[int, int]] = {}

        # metadata
        self._doc_lens: List[int] = []
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

    def _build(self, df: pd.DataFrame) -> None:
        # for each term keep tf per document
        postings: DefaultDict[str, Dict[int, int]] = defaultdict(dict)
        doc_lens: List[int] = []

        texts_pp = df[self.text_pp_col].fillna("").astype(str).tolist()

        for doc_id, text_pp in enumerate(texts_pp):
            tokens = text_pp.split()
            doc_lens.append(len(tokens))

            # term frequencies within document
            tf = Counter(tokens)
            for term, cnt in tf.items():
                postings[term][doc_id] = int(cnt)

        self._index = dict(postings)
        self._vocab = sorted(self._index.keys())
        self._doc_lens = doc_lens

    def get_posting(self, term: str) -> Dict[int, int]:
        """Return posting list for a term: {doc_id: tf}."""
        return self._index.get(term, {})

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_preprocessing: bool = True,
    ) -> pd.DataFrame:
        """
        Search top documents for a query.

        Returns a DataFrame with columns: doc_id, score, text.

        If use_preprocessing=True:
          query is passed through preprocess_text to match the index vocabulary.
        """
        if not query or not str(query).strip():
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        if top_k <= 0:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        #  query to lemmas (same pipeline as for documents)
        if use_preprocessing:
            q_terms = preprocess_text(query)
        else:
            q_terms = str(query).lower().split()

        if not q_terms:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        q_tf = Counter(q_terms)

        # scores per document
        scores: DefaultDict[int, float] = defaultdict(float)

        for term, qcnt in q_tf.items():
            posting = self._index.get(term)
            if not posting:
                continue

            for doc_id, tf in posting.items():
                scores[doc_id] += float(tf) * float(qcnt)

        if not scores:
            return pd.DataFrame(columns=["doc_id", "score", "text"])

        # sort by score desc, then by doc_id asc for stable output
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]

        rows = [
            {"doc_id": doc_id, "score": score, "text": self._texts[doc_id]}
            for doc_id, score in top
        ]
        return pd.DataFrame(rows)

