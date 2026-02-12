from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "movies_merge.csv"

# Load once at import
_new_df = None
_similarity = None


def _load_model():
    global _new_df, _similarity
    if _new_df is not None:
        return
    new_df = pd.read_csv(DATA_PATH)
    ps = PorterStemmer()

    def stem(text):
        return " ".join(ps.stem(w) for w in (text or "").split())

    new_df = new_df.copy()
    new_df["tags"] = new_df["tags"].apply(stem)
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    _new_df = new_df
    _similarity = cosine_similarity(vectors)


def recommend(movie: str, top_k: int = 5):
    _load_model()
    match = _new_df[_new_df["title"] == movie]
    if match.empty:
        return [f"Movie not found: '{movie}'"]
    movie_index = match.index[0]
    distances = _similarity[movie_index]
    ranked = sorted(
        enumerate(distances), key=lambda x: x[1], reverse=True
    )[1 : top_k + 1]
    return [_new_df.iloc[i].title for i, _ in ranked]
