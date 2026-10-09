# Cortex: Universal Agentic Knowledge Assistant

RAG system that answers from **your own documents** (with citations) and
automatically falls back to **web search** when local knowledge is not enough.

## Architecture

```
INGESTION (one time)                    QUERY TIME
data/ (PDF, TXT, MD)                    User question (Streamlit UI)
  -> parse + chunk                        -> hybrid retrieval (vector + BM25, RRF)
  -> embed (MiniLM)                       -> agent check: relevance >= threshold?
  -> ChromaDB                                  yes: use local chunks
                                               no : + DuckDuckGo web search
                                          -> LLM (Gemini / Ollama)
                                          -> answer + numbered citations
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env  -> add GEMINI_API_KEY
```

## Run

```bash
# 1) put PDFs/TXT/MD inside data/   (or upload from the UI sidebar)
python ingest.py
# 2) start the app
streamlit run app.py
# 3) optional: measure retrieval quality
python eval.py
```

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit chat UI |
| `ingest.py` | CLI to build the index |
| `eval.py` + `eval/questions.json` | Hit@k / MRR retrieval evaluation |
| `cortex/ingestion.py` | load, chunk, embed, store |
| `cortex/retriever.py` | hybrid search (dense + BM25 + RRF) |
| `cortex/agent.py` | relevance check + web search fallback |
| `cortex/llm.py` | Gemini / Ollama provider |
| `cortex/pipeline.py` | orchestrates the full answer flow |

## Roadmap
- [ ] Reranker (cross-encoder) after retrieval
- [ ] LLM-based relevance grader instead of a threshold
- [ ] Google Drive / Notion connectors
- [ ] Query rewriting + multi-step reasoning
