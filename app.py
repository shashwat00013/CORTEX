import streamlit as st

from cortex import config
from cortex.ingestion import ingest_all, save_uploads
from cortex.pipeline import answer
from cortex.store import get_collection

st.set_page_config(page_title="Cortex", page_icon="🧠", layout="wide")
st.title("🧠 Cortex")
st.caption("Universal agentic knowledge assistant: your documents first, web when needed.")

with st.sidebar:
    st.header("Knowledge base")
    files = st.file_uploader("Add PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Ingest documents", use_container_width=True):
        if files:
            save_uploads(files)
        with st.spinner("Parsing, chunking, embedding..."):
            stats = ingest_all()
        st.success(f"{stats['files']} files -> {stats['chunks']} chunks")
    try:
        st.metric("Chunks indexed", get_collection().count())
    except Exception:
        st.metric("Chunks indexed", 0)

    st.header("Agent settings")
    allow_web = st.toggle("Web search fallback", value=True)
    use_grader = st.toggle("Smart grader (LLM checks if docs answer it)", value=True)
    threshold = st.slider("Min relevance (below = search web)", 0.0, 1.0, config.MIN_RELEVANCE, 0.05)
    st.caption(f"LLM: {config.LLM_PROVIDER}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])

if q := st.chat_input("Ask something about your documents..."):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + thinking..."):
            try:
                out = answer(q, allow_web=allow_web, threshold=threshold,
                             use_grader=use_grader, history=history)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        st.markdown(out["answer"])
        meta = f"Top similarity: {out['top_similarity']:.2f} | {out['route']}"
        if out["standalone"].strip() != q.strip():
            meta += f" | Understood as: {out['standalone']}"
        st.caption(meta)
        with st.expander("Sources"):
            for i, s in enumerate(out["sources"], 1):
                tag = "WEB" if s["type"] == "web" else "LOCAL"
                st.markdown(f"**[{i}] {tag}** - {s['label']}" + (f" (score {s['score']})" if s["score"] is not None else ""))
                st.write(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))
    st.session_state.messages.append({"role": "assistant", "content": out["answer"], "meta": meta})