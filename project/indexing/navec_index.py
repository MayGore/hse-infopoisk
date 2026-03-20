from pathlib import Path

import pandas as pd
from navec import Navec

from indexing.semantic_base import build_doc_vectors, search_semantic


class NavecIndex:
    """
    Semantic index based on pretrained Navec embeddings.

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
        self.vector_size = self._get_vector_size()

        self._doc_vectors, self._doc_norms = build_doc_vectors(
            df=df,
            get_token_vector=self.get_token_vector,
            vector_size=self.vector_size,
            text_pp_col=self.text_pp_col,
        )

    def _load_model(self, model_path: Path):
        """Load pretrained Navec model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        return Navec.load(str(model_path))

    def _get_vector_size(self) -> int:
        """Detect embedding dimension."""
        try:
            return int(self._model.pq.dim)
        except Exception:
            pass

        sample_tokens = ["и", "в", "на", "дом", "рецепт", "суп"]

        for token in sample_tokens:
            vec = self.get_token_vector(token)
            if vec is not None:
                return int(len(vec))

        raise ValueError("Не удалось определить размер векторов в модели Navec")

    def get_token_vector(self, token: str):
        """Return token vector or None if token is not in the model."""
        try:
            if token in self._model:
                return self._model[token]
        except Exception:
            pass

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