# AskMyDocs — version240424b
import streamlit as st
import fitz
import re
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title="AskMyDocs",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

TOP_K = 3  # hardcoded — no need to expose this to users

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def extract_text(file):
    ext = file.name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return [page.get_text() for page in doc]
    raw = file.read()
    text = (raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip()
    return [text]

def make_chunks(pages, size=300, overlap=50):
    all_chunks = []
    for page_text in pages:
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
                keep, kept = [], 0
                for sent in reversed(buf):
                    keep.insert(0, sent)
                    kept += len(sent.split())
                    if kept >= overlap:
                        break
                buf, count = keep, kept
        if buf:
            leftover = " ".join(buf)
            if all_chunks and len(leftover.split()) < 40:
                all_chunks[-1] = all_chunks[-1] + " " + leftover
            else:
                all_chunks.append(leftover)
    return all_chunks

@st.cache_data(show_spinner="Building semantic index...")
def embed_chunks(chunks_tuple):
    """Cache embeddings by document content so reprocessing is instant."""
    return embedder.encode(list(chunks_tuple), convert_to_tensor=False, show_progress_bar=False)

def retrieve(query, chunks, chunk_embs_np):
    q_emb = embedder.encode(query, convert_to_tensor=False)
    chunk_tensor = util.normalize_embeddings(
        __import__("torch").tensor(chunk_embs_np)
    )
    q_tensor = util.normalize_embeddings(
        __import__("torch").tensor(q_emb).unsqueeze(0)
    )
    scores = (q_tensor @ chunk_tensor.T)[0].numpy()
    idx = np.argsort(scores)[::-1][:TOP_K]
    result_idx = list(idx)
    if 0 not in result_idx and len(chunks) > 0:
        result_idx = [0] + result_idx[:TOP_K - 1]
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
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip(), resp.usage

for k, v in {"chunks": [], "embs": None, "filename": "", "processed": False, "history": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

groq_key = st.secrets.get("GROQ_API_KEY", "")

# ── Sidebar ──────────────────────────────────────────────────────────
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
    st.divider()
    if st.session_state.processed:
        st.markdown(f"**📄 {st.session_state.filename}**")
        st.caption(f"{len(st.session_state.chunks)} chunks indexed")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ── Main ─────────────────────────────────────────────────────────────
st.markdown("# 📄 AskMyDocs")
st.markdown("Upload a document · ask anything · answers come **only** from your document.")

uploaded = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

if uploaded:
    if uploaded.size > 15 * 1024 * 1024:
        st.error("File too large (max 15 MB)."); st.stop()

    if uploaded.name != st.session_state.filename:
        st.session_state.update(chunks=[], embs=None, filename=uploaded.name,
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
            # embed_chunks is cached — fast on repeat, safe on first run
            embs_np = embed_chunks(tuple(chunks))
            st.session_state.update(chunks=chunks, embs=embs_np, processed=True)
            bar.progress(100, "Ready!")
            bar.empty()
            st.success(f"✅ **{uploaded.name}** — {len(chunks)} chunks indexed")
            st.rerun()
    else:
        st.success(f"✅ **{st.session_state.filename}** — {len(st.session_state.chunks)} chunks indexed")
        if st.button("🔄 Reprocess Document", use_container_width=True):
            st.session_state.processed = False
            st.rerun()

# ── Chat ─────────────────────────────────────────────────────────────
if st.session_state.processed:
    st.divider()
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            if "NOT FOUND" in turn["a"].upper():
                st.markdown(f'<span style="color:#e74c3c">{turn["a"]}</span>', unsafe_allow_html=True)
            else:
                st.write(turn["a"])
            if "score" in turn:
                st.caption(f"Similarity: {turn['score']:.2f} · Tokens: {turn.get('tokens', '—')}")

    if prompt := st.chat_input("Ask a question about your document..."):
        if not groq_key:
            st.error("No GROQ_API_KEY found in secrets."); st.stop()

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                top_chunks, score = retrieve(
                    prompt,
                    st.session_state.chunks,
                    st.session_state.embs
                )
                context = pack_context(top_chunks)
                answer, usage = ask_groq(
                    prompt, context,
                    st.session_state.history,
                    groq_key, model, temperature
                )

            if "NOT FOUND" in answer.upper():
                st.markdown(f'<span style="color:#e74c3c">{answer}</span>', unsafe_allow_html=True)
            else:
                st.write(answer)

            tokens = getattr(usage, "total_tokens", "—")
            st.caption(f"Similarity: {score:.2f} · Tokens: {tokens}")

        st.session_state.history.append({
            "q": prompt, "a": answer,
            "score": score, "tokens": tokens
        })
