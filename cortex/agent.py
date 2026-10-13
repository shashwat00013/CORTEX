"""The 'agentic' part: grade context, rewrite queries, search the web."""
import json
import re

from . import config
from .llm import generate


def needs_web(top_similarity, threshold=None):
    """Fast pre-check: very low similarity means local docs are clearly off-topic."""
    threshold = config.MIN_RELEVANCE if threshold is None else threshold
    return top_similarity < threshold


def _parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def condense_question(history, question):
    """Turn a follow-up ('explain more') into a standalone question using chat history."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
    prompt = (
        "Rewrite the user's last message as ONE standalone question that makes sense "
        "without the conversation. Keep the same language. If it is already standalone, "
        "return it unchanged. Output only the question.\n\n"
        f"Conversation:\n{convo}\n\nLast message: {question}\n\nStandalone question:"
    )
    try:
        out = generate(prompt).strip().strip('"')
        return out or question
    except Exception:
        return question


def grade_context(question, chunks):
    """Ask the LLM whether the chunks really answer the question.
    Returns (sufficient: bool, web_search_query: str)."""
    if not chunks:
        return False, question
    ctx = "\n\n".join(f"[{i}] {c['text'][:600]}" for i, c in enumerate(chunks, 1))
    prompt = (
        "You are a strict grader for a retrieval system.\n"
        "Decide whether the context below contains enough information to answer the "
        "question directly. Related-but-not-answering text counts as NOT sufficient.\n"
        "Also write a short, effective web search query for the question.\n"
        'Reply with ONLY JSON: {"sufficient": true or false, "search_query": "..."}\n\n'
        f"Question: {question}\n\nContext:\n{ctx}"
    )
    try:
        data = _parse_json(generate(prompt))
        return bool(data.get("sufficient")), (data.get("search_query") or question)
    except Exception as e:
        print(f"[grader failed, trusting local context] {e}")
        return True, question


def web_search(query, n=4):
    """DuckDuckGo search, no API key needed. Returns [{title, url, text}]."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        hits = DDGS().text(query, max_results=n)
        return [
            {"title": h.get("title", ""), "url": h.get("href", ""), "text": h.get("body", "")}
            for h in hits
        ]
    except Exception as e:
        print(f"[web_search failed] {e}")
        return []