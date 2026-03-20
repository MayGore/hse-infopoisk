from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "data_short_pp.csv"
MODEL_DIR = ROOT_DIR / "models" / "word2vec"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

sentences = (
    df["text_pp"]
    .fillna("")
    .astype(str)
    .str.split()
    .tolist()
)

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,       # skip-gram
    epochs=20,
)

model.wv.save(str(MODEL_DIR / "model.kv"))
print("Model saved to", MODEL_DIR / "model.kv")