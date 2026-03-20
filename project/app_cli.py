from pathlib import Path
import argparse
import time

import pandas as pd

from data.loader import load_short
from indexing.bm25_matrix import BM25MatrixIndex
from indexing.word2vec_index import Word2VecIndex
from indexing.navec_index import NavecIndex
from utils.input_validation import detect_language
from utils.preprocessing import get_or_create_preprocessed


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

PP_PATH = DATA_DIR / "data_short_pp.csv"

MODEL_PATHS = {
    "word2vec": MODELS_DIR / "word2vec" / "model.kv",
    "navec": MODELS_DIR / "navec" / "model.tar",
}


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Поиск по коллекции документов: BM25, word2vec, Navec"
    )

    parser.add_argument(
        "--query",
        required=True,
        help='Текст запроса. Пример: --query "полезные сладости"',
    )
    parser.add_argument(
        "--index",
        default="bm25",
        choices=["bm25", "word2vec", "navec"],
        help="Тип индекса: bm25, word2vec, navec",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Сколько документов вывести",
    )

    return parser.parse_args()


def build_index(df_pp: pd.DataFrame, index_name: str):
    """Create index instance according to user choice."""
    if index_name == "bm25":
        return BM25MatrixIndex(df_pp)

    if index_name == "word2vec":
        return Word2VecIndex(
            df=df_pp,
            model_path=MODEL_PATHS["word2vec"],
        )

    if index_name == "navec":
        return NavecIndex(
            df=df_pp,
            model_path=MODEL_PATHS["navec"],
        )

    raise ValueError("Неподдерживаемый тип индекса")


def print_hits(df: pd.DataFrame, snippet: int = 160) -> None:
    """Pretty-print search results."""
    for i, row in enumerate(df.itertuples(index=False), 1):
        doc_id = getattr(row, "doc_id")
        score = getattr(row, "score")
        text = str(getattr(row, "text")).replace("\n", " ").strip()

        if len(text) > snippet:
            text = text[: snippet - 1] + "…"

        print(f"{i:>2}. doc_id={doc_id}  score={score:.4f}  {text}")


def run_search(query: str, index_name: str = "bm25", top_k: int = 10) -> pd.DataFrame:
    """
    Run search programmatically.

    Loads dataset, ensures preprocessing exists, builds index and returns top results.
    """
    if not query or not str(query).strip():
        raise ValueError("Запрос пустой")

    if detect_language(query) != "rus":
        raise ValueError("Запрос должен быть на русском языке")

    if top_k <= 0:
        raise ValueError("top_k должен быть > 0")

    df_raw = load_short(auto_create=False)
    df_pp = get_or_create_preprocessed(df_raw, PP_PATH)

    index = build_index(df_pp, index_name=index_name)

    start = time.perf_counter()
    res = index.search(query=query, top_k=top_k, use_preprocessing=True)
    elapsed = time.perf_counter() - start

    print(f"Индекс: {index_name}")
    print(f"Время поиска: {elapsed:.4f} сек.")

    return res


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 200)
    pd.set_option("display.width", 0)
    pd.set_option("display.expand_frame_repr", False)

    res = run_search(
        query=args.query,
        index_name=args.index,
        top_k=args.top_k,
    )

    if res.empty:
        print("Ничего не найдено.")
        return

    res = res.copy()
    res["score"] = res["score"].round(4)
    print_hits(res)


if __name__ == "__main__":
    main()
