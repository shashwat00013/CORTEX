"""End-to-end RAG: retrieve -> agent check -> (web search) -> LLM -> answer + sources."""
from . import config
from .agent import needs_web, web_search
from .llm import generate
from .retriever import retrieve

SYSTEM = (
    "You are Cortex, a careful knowledge assistant. Answer ONLY from the numbered "
    "context. Cite sources inline like [1], [2]. If the context does not contain "
    "the answer, say you could not find it. Do not invent facts."
)


def answer(question, allow_web=True, threshold=None, k=config.TOP_K):
    chunks, top_sim = retrieve(question, k)

    sources = [
        {
            "type": "local",
            "label": f"{c['source']} (page {c['page']})",
            "text": c["text"],
            "score": round(c["similarity"], 3),
        }
        for c in chunks
    ]

    used_web = False
    if allow_web and needs_web(top_sim, threshold):
        for h in web_search(question):
            sources.append(
                {"type": "web", "label": f"{h['title']} - {h['url']}", "text": h["text"], "score": None}
            )
        used_web = any(s["type"] == "web" for s in sources)

    if not sources:
        return {
            "answer": "Mujhe kuch relevant nahi mila. Pehle documents ingest karo ya web search on karo.",
            "sources": [],
            "used_web": False,
            "top_similarity": top_sim,
        }

    context = "\n\n".join(f"[{i}] ({s['label']})\n{s['text']}" for i, s in enumerate(sources, 1))
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"
    return {
        "answer": generate(prompt, SYSTEM),
        "sources": sources,
        "used_web": used_web,
        "top_similarity": top_sim,
    }
