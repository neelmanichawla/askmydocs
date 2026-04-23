import streamlit as st
import fitz
import re
import numpy as np
import nltk
from groq import Groq

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

CHUNK_MAX_WORDS     = 150
CHUNK_OVERLAP_SENTS = 3
RETRIEVE_TOP_K      = 8
RERANK_TOP_K        = 5
GROQ_MODEL          = "llama-3.1-8b-instant"

@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    retrieval = SentenceTransformer('BAAI/bge-small-en-v1.5')
    reranker  = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return retrieval, reranker

retrieval_model, reranker_model = load_models()

def extract_text(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            return "".join(p.get_text() for p in doc).strip()
        except Exception as e:
            st.error(f"PDF error: {e}")
            return ""
    elif ext == 'txt':
        try:
            content = uploaded_file.read()
            return (content.decode('utf-8') if hasattr(content, 'decode') else content).strip()
        except Exception as e:
            st.error(f"Text file error: {e}")
            return ""
    else:
        st.error("Please upload a PDF or TXT file.")
        return ""

def chunk_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = sent_tokenize(text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        current.append(sent)
        count += len(sent.split())
        if count >= CHUNK_MAX_WORDS:
            chunks.append(" ".join(current))
            current = current[-CHUNK_OVERLAP_SENTS:]
            count = sum(len(s.split()) for s in current)
    if current:
        chunks.append(" ".join(current))
    return chunks

def build_index(chunks):
    return retrieval_model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)

def retrieve(query, chunks, embeddings):
    from sentence_transformers import util
    from rank_bm25 import BM25Okapi

    q_vec       = retrieval_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    sem_scores  = util.cos_sim(q_vec, embeddings).cpu().numpy()[0]

    tokenized   = [c.lower().split() for c in chunks]
    bm25        = BM25Okapi(tokenized)
    bm25_scores = np.array(bm25.get_scores(query.lower().split()))

    sem_ranks  = np.argsort(sem_scores)[::-1]
    bm25_ranks = np.argsort(bm25_scores)[::-1]

    rrf = np.zeros(len(chunks))
    for rank, idx in enumerate(sem_ranks):
        rrf[idx] += 1.0 / (rank + 60)
    for rank, idx in enumerate(bm25_ranks):
        rrf[idx] += 1.0 / (rank + 60)

    top_idx    = np.argsort(rrf)[::-1][:RETRIEVE_TOP_K]
    top_chunks = [chunks[i] for i in top_idx]
    best_score = float(sem_scores[top_idx[0]]) if len(top_idx) > 0 else 0.0

    return top_chunks, best_score, sem_scores

def rerank(query, chunks):
    if not chunks:
        return [], []
    scores = reranker_model.predict([(query, c) for c in chunks])
    ranked = sorted(zip(scores, chunks), reverse=True)[:RERANK_TOP_K]
    return [c for _, c in ranked], [float(s) for s, _ in ranked]

def is_relevant(best_score, all_scores, base=0.1):
    if len(all_scores) == 0:
        return False
    threshold = max(base, float(np.mean(all_scores)) + 0.5 * float(np.std(all_scores)))
    return best_score >= threshold

def generate_answer(question, chunks):
    if not chunks:
        return "NOT FOUND IN DOCUMENT"
    context = "\n\n".join(chunks)
    prompt  = f"""You are a precise document analyst.

RULES:
- Answer ONLY using the context below
- Be concise and direct
- If asked to summarize, give 3-5 key bullet points
- If the answer is not in the context, say exactly: NOT FOUND IN DOCUMENT
- Do not use outside knowledge

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    try:
        client   = Groq(api_key=st.secrets["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Groq error: {e}")
        return ""

def main():
    st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="centered")
    st.title("📄 AskMyDocs")
    st.markdown("Upload a document and ask anything about it.")

    with st.sidebar:
        st.header("How it works")
        st.markdown("""
1. Upload a PDF or TXT file
2. Ask any question
3. Get an answer grounded only in your document

**Privacy:** Your document is processed in memory only.
        """)
        st.markdown("---")
        st.caption("Built with BGE embeddings + BM25 + Groq LLM")

    for key, val in [("processed", False), ("chunks", []),
                     ("embeddings", None), ("last_file", "")]:
        if key not in st.session_state:
            st.session_state[key] = val

    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.last_file:
            st.session_state.processed  = False
            st.session_state.chunks     = []
            st.session_state.embeddings = None
            st.session_state.last_file  = uploaded_file.name

        if uploaded_file.size > 15 * 1024 * 1024:
            st.error("File too large. Maximum 15MB.")
            st.stop()

        if not st.session_state.processed or st.button("🔄 Reprocess"):
            uploaded_file.seek(0)
            with st.spinner("Reading document..."):
                text = extract_text(uploaded_file)
                if not text:
                    st.error("Could not extract text.")
                    st.stop()
            with st.spinner("Processing..."):
                chunks = chunk_text(text)
                if not chunks:
                    st.error("Could not create chunks.")
                    st.stop()
                st.session_state.chunks     = chunks
                st.session_state.embeddings = build_index(chunks)
                st.session_state.processed  = True
            st.success(f"✅ Ready — {len(chunks)} chunks indexed")

    if st.session_state.processed:
        st.divider()
        question = st.text_input("Ask a question:", placeholder="What is this document about?")
        if question:
            with st.spinner("Searching..."):
                top_chunks, best_score, all_scores = retrieve(
                    question, st.session_state.chunks, st.session_state.embeddings
                )
                reranked, _ = rerank(question, top_chunks)
            if not is_relevant(best_score, all_scores):
                st.warning("This question doesn't seem to match the document content.")
            else:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, reranked)
                if answer:
                    st.success("**Answer:**")
                    st.write(answer)
                with st.expander("View source chunks"):
                    for i, chunk in enumerate(reranked):
                        st.markdown(f"**Chunk {i+1}**")
                        st.caption(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.divider()
    elif not uploaded_file:
        st.info("👆 Upload a PDF or TXT file to get started.")

if __name__ == "__main__":
    main()
