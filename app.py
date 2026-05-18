# AskMyDocs
import streamlit as st
import fitz
import re
import numpy as np
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(
    page_title="AskMyDocs",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

TOP_K = 3
CANDIDATE_K = 12
RRF_K = 60
MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.1

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
    for page_num, page_text in enumerate(pages, start=1):
        page_text = re.sub(r"\s+", " ", page_text).strip()
        if not page_text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", page_text)
        buf, count = [], 0
        for s in sentences:
            buf.append(s)
            count += len(s.split())
            if count >= size:
                all_chunks.append({
                    "text": " ".join(buf),
                    "page": page_num,
                    "chunk_index": len(all_chunks),
                })
                keep, kept = [], 0
                for sent in reversed(buf):
                    keep.insert(0, sent)
                    kept += len(sent.split())
                    if kept >= overlap:
                        break
                buf, count = keep, kept
        if buf:
            leftover = " ".join(buf)
            if all_chunks and all_chunks[-1]["page"] == page_num and len(leftover.split()) < 40:
                all_chunks[-1]["text"] = all_chunks[-1]["text"] + " " + leftover
            else:
                all_chunks.append({
                    "text": leftover,
                    "page": page_num,
                    "chunk_index": len(all_chunks),
                })
    return all_chunks

@st.cache_data(show_spinner="Building semantic index...")
def embed_chunks(chunks_tuple):
    return embedder.encode(list(chunks_tuple), convert_to_tensor=False, show_progress_bar=False)

def format_sources(chunks):
    return [
        f"{c['filename']}, page {c['page']}, chunk {c['chunk_index'] + 1}"
        for c in chunks
    ]

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def rrf_scores(rankings, k=RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores

def retrieve(query, chunks, chunk_embs_np):
    q_emb = embedder.encode(query, convert_to_tensor=False)
    chunk_tensor = util.normalize_embeddings(torch.tensor(chunk_embs_np))
    q_tensor = util.normalize_embeddings(torch.tensor(q_emb).unsqueeze(0))
    scores = (q_tensor @ chunk_tensor.T)[0].numpy()
    candidate_k = min(CANDIDATE_K, len(chunks))
    dense_idx = list(np.argsort(scores)[::-1][:candidate_k])

    tokenized_chunks = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_idx = list(np.argsort(bm25_scores)[::-1][:candidate_k])

    fused = rrf_scores([dense_idx, bm25_idx])
    result_idx = [
        idx for idx, _ in sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True
        )[:TOP_K]
    ]
    return [chunks[i] for i in result_idx], float(scores[dense_idx[0]])

def pack_context(chunks, token_budget=2800):
    char_budget = token_budget * 4
    parts, total = [], 0
    for c in chunks:
        if total + len(c["text"]) > char_budget:
            break
        parts.append(c["text"])
        total += len(c["text"])
    return "\n\n---\n\n".join(parts)

def ask_groq(question, context, history, api_key):
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
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip(), resp.usage

for k, v in {"chunks": [], "embs": None, "filename": "", "processed": False, "history": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

groq_key = st.secrets.get("GROQ_API_KEY", "")

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Document")
    st.divider()
    if st.session_state.processed:
        st.markdown(f"**{st.session_state.filename}**")
        st.caption(f"{len(st.session_state.chunks)} chunks indexed")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No document loaded yet.")

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
            for c in chunks:
                c["filename"] = uploaded.name
            embs_np = embed_chunks(tuple(c["text"] for c in chunks))
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
            if turn.get("sources"):
                st.caption("Sources: " + "; ".join(turn["sources"]))

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
                sources = format_sources(top_chunks)
                answer, usage = ask_groq(
                    prompt, context,
                    st.session_state.history,
                    groq_key
                )

            if "NOT FOUND" in answer.upper():
                st.markdown(f'<span style="color:#e74c3c">{answer}</span>', unsafe_allow_html=True)
            else:
                st.write(answer)

            tokens = getattr(usage, "total_tokens", "—")
            st.caption(f"Similarity: {score:.2f} · Tokens: {tokens}")
            st.caption("Sources: " + "; ".join(sources))

        st.session_state.history.append({
            "q": prompt, "a": answer,
            "score": score, "tokens": tokens, "sources": sources
        })
