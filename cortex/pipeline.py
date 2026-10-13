"""End-to-end agentic RAG: condense -> retrieve -> grade -> (web search) -> LLM -> answer."""
from datetime import datetime

from . import config
from .agent import condense_question, grade_context, needs_web, web_search
from .llm import generate
from .retriever import retrieve

SYSTEM = (
    "You are Cortex, a careful knowledge assistant. Answer ONLY from the numbered "
    "context and from the current date/time given in the prompt. Cite sources "
    "inline like [1], [2]. If the context answers only part of the question, answer "
    "that part and say clearly what is missing. If it contains nothing relevant, say "
    "you could not find it. Do not invent facts."
)


def answer(question, allow_web=True, threshold=None, k=config.TOP_K,
           use_grader=True, history=None):
    standalone = condense_question(history, question)
    chunks, top_sim = retrieve(standalone, k)

    sources = [
        {
            "type": "local",
            "label": f"{c['source']} (page {c['page']})",
            "text": c["text"],
            "score": round(c["similarity"], 3),
        }
        for c in chunks
    ]

    route = "Answered from local docs"
    web_query = standalone
    go_web = False
    if allow_web:
        if needs_web(top_sim, threshold):
            go_web, route = True, "Web search: local docs not relevant"
        elif use_grader:
            ok, web_query = grade_context(standalone, chunks)
            if not ok:
                go_web, route = True, "Web search: local docs did not answer it"

    used_web = False
    if go_web:
        for h in web_search(web_query):
            sources.append(
                {"type": "web", "label": f"{h['title']} - {h['url']}", "text": h["text"], "score": None}
            )
        used_web = any(s["type"] == "web" for s in sources)
        if not used_web:
            route += " (web returned nothing)"

    now = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    context = "\n\n".join(f"[{i}] ({s['label']})\n{s['text']}" for i, s in enumerate(sources, 1))
    prompt = (
        f"Current date and time: {now}\n\n"
        f"Context:\n{context if context else '(no context found)'}\n\n"
        f"Question: {standalone}\n\nAnswer with citations where context is used:"
    )
    return {
        "answer": generate(prompt, SYSTEM),
        "sources": sources,
        "used_web": used_web,
        "top_similarity": top_sim,
        "route": route,
        "standalone": standalone,
    }