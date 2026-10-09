"""Tiny retrieval evaluation.  python eval.py
A question is a HIT if any expected keyword appears in the top-k chunks.
Edit eval/questions.json with 15-30 questions from YOUR documents."""
import json
from pathlib import Path

from cortex.retriever import retrieve

K = 5

if __name__ == "__main__":
    qs = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
    hits, rr = 0, 0.0
    for item in qs:
        results, _ = retrieve(item["question"], K)
        rank = next(
            (r for r, c in enumerate(results, 1)
             if any(kw.lower() in c["text"].lower() for kw in item["keywords"])),
            None,
        )
        hits += rank is not None
        rr += 1 / rank if rank else 0
        print(f"{'HIT ' if rank else 'MISS'} rank={rank}  {item['question'][:70]}")
    n = len(qs)
    print(f"\nHit@{K}: {hits}/{n} = {hits / n:.0%}   MRR: {rr / n:.2f}")
