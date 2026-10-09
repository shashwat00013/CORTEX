"""Hybrid retrieval: dense (Chroma) + keyword (BM25), fused with RRF."""
import re

from . import config
from .store import embed, get_collection

RRF_K = 60
_cache = {"count": -1, "bm25": None, "ids": [], "docs": [], "metas": []}


def _tok(text):
    return re.findall(r"\w+", text.lower())


def _bm25(col):
    n = col.count()
    if n == 0:
        return None
    if _cache["count"] != n:
        from rank_bm25 import BM25Okapi
        data = col.get(include=["documents", "metadatas"])
        _cache.update(
            count=n,
            ids=data["ids"],
            docs=data["documents"],
            metas=data["metadatas"],
            bm25=BM25Okapi([_tok(d) for d in data["documents"]]),
        )
    return _cache["bm25"]


def rrf(rank_lists, k=RRF_K):
    """Reciprocal Rank Fusion. rank_lists = [[id, id, ...], ...] -> {id: score}."""
    scores = {}
    for ranks in rank_lists:
        for pos, _id in enumerate(ranks):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + pos + 1)
    return scores


def retrieve(query, k=config.TOP_K):
    """Returns (results, top_similarity).
    top_similarity = best dense cosine similarity (0-1) -> used as confidence."""
    col = get_collection()
    n = col.count()
    if n == 0:
        return [], 0.0

    pool = min(k * 3, n)
    res = col.query(
        query_embeddings=embed([query]),
        n_results=pool,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = res["ids"][0]
    sims = {i: 1.0 - d for i, d in zip(dense_ids, res["distances"][0])}
    store = {
        i: (doc, meta)
        for i, doc, meta in zip(dense_ids, res["documents"][0], res["metadatas"][0])
    }

    bm25_ids = []
    bm25 = _bm25(col)
    if bm25 is not None:
        scores = bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for i in order[:pool]:
            if scores[i] > 0:
                bm25_ids.append(_cache["ids"][i])
                store.setdefault(
                    _cache["ids"][i], (_cache["docs"][i], _cache["metas"][i])
                )

    fused = rrf([dense_ids, bm25_ids])
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    results = [
        {
            "id": i,
            "text": store[i][0],
            "source": store[i][1]["source"],
            "page": store[i][1]["page"],
            "similarity": sims.get(i, 0.0),
        }
        for i in top
    ]
    top_sim = max(sims.values()) if sims else 0.0
    return results, top_sim
