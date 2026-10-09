"""Run once after adding files to data/:  python ingest.py"""
from cortex.ingestion import ingest_all

if __name__ == "__main__":
    stats = ingest_all()
    print(f"Done. {stats['files']} files -> {stats['chunks']} chunks indexed.")
