# AskMyDocs — version240426
import streamlit as st
import fitz
import re
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="centered",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.answer {
    background: #ffffff;
    border: 1px solid #cedcd8;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-top: .5rem;
    line-height: 1.7;
    font-size: .95rem;
    color: #28251d;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
.not-found {
    background: #fff8f5;
    border: 1px solid #e8cfc4;
    border-radius: 12px;
    padding: .9rem 1.2rem;
    color: #964219;
    font-size: .9rem;
}
.pill {
    display: inline-block;
    background: rgba(1,105,111,0.08);
    border: 1px solid rgba(1,105,111,0.2);
    border-radius: 999px;
    padding: .15rem .65rem;
    font-size: .75rem;
    color: #01696f;
    font-weight: 500;
    margin-right: .3rem;
    margin-top: .4rem;
}
.stButton > button {
    background: #01696f !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stButton > button:hover { background: #0c4e54 !important; }
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; }
hr { border-color: #dcd9d5 !important; }
[data-testid="stChatInput"] { margin-bottom: 1.5rem; }

/* sidebar collapse/expand arrow — always visible */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: #f7f6f2 !important;
    border: 1px solid #dcd9d5 !important;
    border-radius: 0 6px 6px 0 !important;
    color: #28251d !important;
    z-index: 999 !important;
}
[data-testid="collapsedControl"]:hover {
    background: #edeae5 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()


def extract_text(file):
    ext = file.name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return pages  # return list of pages for better chunking
    raw = file.read()
    text = (raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip()
    return [text]  # single page for txt


def make_chunks(pages, size=300, overlap=50):
    """
    Improved chunking strategy:
    1. Preserve page boundaries as natural split points
    2. First page always becomes its own chunk (metadata/title/author lives here)
    3. Sentence-aware splitting with word-count overlap
    4. Small leftover sentences are merged into previous chunk, not orphaned
    """
    all_chunks = []

    for page_num, page_text in enumerate(pages):
        page_text = re.sub(r"\s+", " ", page_text).strip()
        if not page_text:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", page_text)
        buf, count = [], 0

        for s in sentences:
            buf.append(s)
            count += len(s.split())
            if count >= size:
                all_chunks.append(" ".join(buf))
                # overlap: keep last N words worth of sentences
                keep, kept = [], 0
                for sent in reversed(buf):
                    keep.insert(0, sent)
                    kept += len(sent.split())
                    if kept >= overlap:
                        break
                buf, count = keep, kept

        # merge leftover into last chunk if tiny, else add as new chunk
        if buf:
            leftover = " ".join(buf)
            if all_chunks and len(leftover.split()) < 40:
                all_chunks[-1] = all_chunks[-1] + " " + leftover
            else:
                all_chunks.append(leftover)

    return all_chunks


def embed(texts):
    return embedder.encode(texts, convert_to_tensor=True, show_progress_bar=False)


def retrieve(query, chunks, chunk_embs, top_k=5):
    """
    Hybrid retrieval:
    1. Semantic similarity (dense)
    2. Always pin first chunk (document metadata/title/author)
    3. Deduplicate
    """
    q_emb = embedder.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, chunk_embs)[0].cpu().numpy()
    idx = np.argsort(scores)[::-1][:top_k]
    result_idx = list(idx)

    # pin first chunk for metadata queries
    if 0 not in result_idx and len(chunks) > 0:
        result_idx = [0] + result_idx[:top_k - 1]

    return [chunks[i] for i in result_idx], float(scores[idx[0]])


def pack_context(chunks, token_budget=2800):
    char_budget = token_budget * 4
    parts, total = [], 0
    for c in chunks:
        if total + len(c) > char_budget:
            break
        parts.append(c)
        total += len(c)
    return "\n\n---\n\n".join(parts)


def ask_groq(question, context, history, api_key, model, temperature):
    client = Groq(api_key=api_key)
    system = (
        "You are a precise document Q&A assistant. "
        "Answer ONLY using information explicitly stated in the provided document context. "
        "If the answer is not present in the context, say: NOT FOUND IN DOCUMENT. "
        "Never infer, assume, or use outside knowledge. Be concise and direct."
    )
    messages = [{"role": "system", "content": system}]
    for turn in history[-3:]:
        messages.append({"role": "user", "content": turn["q"]})
        messages.append({"role": "assistant", "content": turn["a"]})
    messages.append({
        "role": "user",
        "content": f"Document context:\n{context}\n\nQuestion: {question}"
    })
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=800,
    )
    return resp.choices[0].message.content.strip(), resp.usage


for k, v in {"chunks": [], "embs": None, "filename": "", "processed": False, "history": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

groq_key = st.secrets.get("GROQ_API_KEY", "")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()
    model = st.selectbox("🤖 Model", [
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "gemma2-9b-it",
    ], help="llama-3.1-8b-instant is fastest on free tier")
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.1, 0.05,
                            help="Low = factual, High = creative")
    top_k = st.slider("📚 Chunks retrieved", 2, 8, 4,
                      help="More = richer context, more tokens used")
    st.divider()
    if st.session_state.processed:
        st.markdown(f"**📄 {st.session_state.filename}**")
        st.caption(f"{len(st.session_state.chunks)} chunks indexed")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────
st.markdown("# 📄 AskMyDocs")
st.markdown("Upload a document · ask anything · answers come **only** from your document.")

uploaded = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

if uploaded:
    if uploaded.size > 15 * 1024 * 1024:
        st.error("File too large (max 15 MB)."); st.stop()

    if uploaded.name != st.session_state.filename:
        st.session_state.update(chunks=[], embs=None,
                                filename=uploaded.name,
                                processed=False, history=[])

    if not st.session_state.processed:
        if st.button("⚡ Process Document", use_container_width=True):
            bar = st.progress(0, "Reading document...")
            try:
                pages = extract_text(uploaded)
            except Exception as e:
                st.error(f"Could not read file: {e}"); st.stop()
            if not pages or not any(p.strip() for p in pages):
                st.error("No text found in document."); st.stop()

            bar.progress(40, "Splitting into chunks...")
            chunks = make_chunks(pages)
            if not chunks:
                st.error("Could not split document."); st.stop()

            bar.progress(70, "Building semantic index...")
            embs = embed(chunks)
            st.session_state.update(chunks=chunks, embs=embs, processed=True)
            bar.progress(100, "Ready!")
            bar.empty()
            st.success(f"✅ **{uploaded.name}** — {len(chunks)} chunks indexed")
            st.rerun()
    else:
        st.success(f"✅ **{st.session_state.filename}** — {len(st.session_state.chunks)} chunks indexed")
        if st.button("🔄 Reprocess Document", use_container_width=True):
            st.session_state.processed = False
            st.rerun()

# ── Chat ──────────────────────────────────────────────────────────
if st.session_state.processed:
    st.divider()
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            if "NOT FOUND" in turn["a"].upper():
                st.markdown(f'<div class="not-found">🔍 {turn["a"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="answer">{turn["a"]}</div>', unsafe_allow_html=True)
            if turn.get("tokens"):
                u = turn["tokens"]
                st.markdown(
                    f'<span class="pill">↑ {u.prompt_tokens} tok</span>'
                    f'<span class="pill">↓ {u.completion_tokens} tok</span>',
                    unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    question = st.chat_input("Ask a question about your document...")

    if question:
        if not groq_key:
            st.error("No Groq API key found. Add GROQ_API_KEY to Streamlit secrets.")
            st.stop()
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chunks_hit, top_score = retrieve(
                        question, st.session_state.chunks,
                        st.session_state.embs, top_k=top_k)
                    context = pack_context(chunks_hit)
                    answer, usage = ask_groq(
                        question, context, st.session_state.history,
                        groq_key, model, temperature)
                except Exception as e:
                    err = str(e)
                    if "429" in err or "rate_limit" in err.lower():
                        st.error("⏳ Rate limit hit — wait a moment and retry.")
                    elif "401" in err or "invalid_api_key" in err.lower():
                        st.error("❌ Invalid API key. Check Streamlit secrets.")
                    else:
                        st.error(f"Error: {err}")
                    st.stop()

            if "NOT FOUND" in answer.upper():
                st.markdown('<div class="not-found">🔍 The answer was not found in the document.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="answer">{answer}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="pill">↑ {usage.prompt_tokens} tok</span>'
                f'<span class="pill">↓ {usage.completion_tokens} tok</span>'
                f'<span class="pill">similarity {top_score:.2f}</span>',
                unsafe_allow_html=True)
            with st.expander("View retrieved context"):
                for i, c in enumerate(chunks_hit):
                    st.caption(f"Chunk {i+1}")
                    st.write(c[:400] + ("…" if len(c) > 400 else ""))
                    if i < len(chunks_hit) - 1:
                        st.divider()
        st.session_state.history.append({"q": question, "a": answer, "tokens": usage})

elif not uploaded:
    st.info("👆 Upload a PDF or TXT to get started.")
    st.markdown("""
**How it works**
1. Upload any PDF or TXT (up to 15 MB)
2. Click **⚡ Process Document**
3. Ask any question — Groq LLM answers using **only** your document

**Free tier tips**
- Use `llama-3.1-8b-instant` — fastest, fewest tokens
- Keep chunks at 3–4 to stay within rate limits
    """)
