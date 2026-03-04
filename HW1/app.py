from pathlib import Path

import pandas as pd

from data.loader import load_short
from indexing.bm25_dict import BM25DictIndex
from indexing.bm25_matrix import BM25MatrixIndex
from indexing.freq_dict import FreqDictIndex
from indexing.freq_matrix import FreqMatrixIndex
from utils.input_validation import read_user_input
from utils.preprocessing import get_or_create_preprocessed


def build_index(df_pp: pd.DataFrame, index_type: str, impl: str):
    """Create an index instance according to user choice."""
    if index_type == "F" and impl == "dict":
        return FreqDictIndex(df_pp)
    if index_type == "F" and impl == "matrix":
        return FreqMatrixIndex(df_pp)
    if index_type == "BM" and impl == "dict":
        return BM25DictIndex(df_pp)
    if index_type == "BM" and impl == "matrix":
        return BM25MatrixIndex(df_pp)

    raise ValueError("Неподдерживаемая комбинация индекса и реализации")


def print_hits(df, snippet=160, top=None) -> None:
    if top is not None:
        df = df.head(top)

    for i, row in enumerate(df.itertuples(index=False), 1):
        doc_id = getattr(row, "doc_id")
        score = getattr(row, "score")
        text = str(getattr(row, "text")).replace("\n", " ").strip()
        if len(text) > snippet:
            text = text[:snippet - 1] + "…"
        print(f"{i:>2}. doc_id={doc_id}  score={score:.4f}  {text}")


def run_search(query: str, index_type: str = "BM", impl: str = "dict", top_k: int = 5) -> pd.DataFrame:
    """
    Entry point for running search as a programmer.

    Loads short dataset, ensures preprocessed file exists, builds an index, returns top_k results.
    """
    df_raw = load_short(auto_create=False)
    pp_path = Path(__file__).resolve().parent / "data" / "data_short_pp.csv"
    df_pp = get_or_create_preprocessed(df_raw, pp_path)

    index = build_index(df_pp, index_type=index_type, impl=impl)
    return index.search(query, top_k=top_k, use_preprocessing=True)  # mquery also goes through preprocessing


def run_demo() -> None:
    """Interactive demo: read input, build index, print top results."""
    query, index_type, impl = read_user_input()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 200)
    pd.set_option("display.width", 0)
    pd.set_option("display.expand_frame_repr", False)

    res = run_search(query=query, index_type=index_type, impl=impl, top_k=10)
    if res.empty:
        print("Ничего не найдено.")
        return

    res = res.copy()
    res["score"] = res["score"].round(4)
    print_hits(res, top=10)


if __name__ == "__main__":
    run_demo()
