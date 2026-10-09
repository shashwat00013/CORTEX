"""Embeddings + Chroma vector DB. Heavy imports are lazy so startup stays fast."""
from functools import lru_cache
from . import config


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts):
    return get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


@lru_cache(maxsize=1)
def _client():
    import chromadb
    config.DB_DIR.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(config.DB_DIR))


def get_collection():
    return _client().get_or_create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def reset_collection():
    try:
        _client().delete_collection(config.COLLECTION)
    except Exception:
        pass
    return get_collection()
