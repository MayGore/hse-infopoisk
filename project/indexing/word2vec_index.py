from pathlib import Path

import pandas as pd
from gensim.models import KeyedVectors

from indexing.semantic_base import build_doc_vectors, search_semantic


class Word2VecIndex:
    """
    Semantic index based on pretrained word2vec embeddings.

    - document vector = mean of token vectors
    - query vector = mean of token vectors
    - similarity = cosine similarity
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_path: str | Path,
        text_pp_col: str = "text_pp",
        text_col: str = "text",
    ) -> None:
        self.text_pp_col = text_pp_col
        self.text_col = text_col
        self.model_path = Path(model_path)

        if text_pp_col not in df.columns or text_col not in df.columns:
            raise ValueError(f"В датафрейме должны быть колонки '{text_pp_col}' и '{text_col}'")

        self._texts = df[text_col].astype(str).tolist()
        self._model = self._load_model(self.model_path)
        self.vector_size = int(self._model.vector_size)

        self._doc_vectors, self._doc_norms = build_doc_vectors(
            df=df,
            get_token_vector=self.get_token_vector,
            vector_size=self.vector_size,
            text_pp_col=self.text_pp_col,
        )

    def _load_model(self, model_path: Path) -> KeyedVectors:
        """Load pretrained word2vec model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        suffix = model_path.suffix.lower()

        if suffix in {".kv", ".model"}:
            return KeyedVectors.load(str(model_path))

        if suffix in {".bin", ".txt", ".vec"}:
            binary = suffix == ".bin"
            return KeyedVectors.load_word2vec_format(str(model_path), binary=binary)

        raise ValueError(
            "Неподдерживаемый формат модели. "
            "Ожидался один из: .kv, .model, .bin, .txt, .vec"
        )

    def get_token_vector(self, token: str):
        """Return token vector or None if token is not in the model."""
        if token in self._model.key_to_index:
            return self._model[token]
        return None

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_preprocessing: bool = True,
    ) -> pd.DataFrame:
        """Search top documents for a query."""
        return search_semantic(
            query=query,
            texts=self._texts,
            doc_vectors=self._doc_vectors,
            doc_norms=self._doc_norms,
            get_token_vector=self.get_token_vector,
            vector_size=self.vector_size,
            top_k=top_k,
            use_preprocessing=use_preprocessing,
        )