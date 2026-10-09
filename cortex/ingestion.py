"""Ingestion pipeline: load files -> chunk -> embed -> store in vector DB."""
import re
from pathlib import Path

from . import config
from .store import embed, reset_collection

SUPPORTED = {".pdf", ".txt", ".md"}


def load_documents(data_dir: Path = config.DATA_DIR):
    """Yield {source, page, text} for every supported file."""
    for path in sorted(Path(data_dir).glob("**/*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    yield {"source": path.name, "page": i, "text": text}
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                yield {"source": path.name, "page": 1, "text": text}


def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Fixed-size chunks with overlap, cut at word boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def save_uploads(files):
    """Save Streamlit uploaded files into data/."""
    config.DATA_DIR.mkdir(exist_ok=True)
    for f in files:
        (config.DATA_DIR / f.name).write_bytes(f.getbuffer())


def ingest_all():
    """Rebuild the whole index from data/. Returns {'files': n, 'chunks': n}."""
    ids, docs, metas, sources = [], [], [], set()
    for d in load_documents():
        sources.add(d["source"])
        for i, chunk in enumerate(chunk_text(d["text"])):
            ids.append(f"{d['source']}::p{d['page']}::c{i}")
            docs.append(chunk)
            metas.append({"source": d["source"], "page": d["page"]})

    col = reset_collection()
    for s in range(0, len(docs), 64):
        batch = slice(s, s + 64)
        col.add(
            ids=ids[batch],
            documents=docs[batch],
            metadatas=metas[batch],
            embeddings=embed(docs[batch]),
        )
    return {"files": len(sources), "chunks": len(docs)}
