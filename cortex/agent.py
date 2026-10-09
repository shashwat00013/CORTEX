"""The 'agentic' part: decide if local context is enough, else search the web."""
from . import config


def needs_web(top_similarity, threshold=None):
    """Simple decision rule. Upgrade later: ask the LLM to grade the chunks."""
    threshold = config.MIN_RELEVANCE if threshold is None else threshold
    return top_similarity < threshold


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
