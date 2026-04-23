import streamlit as st
import fitz
import numpy as np
import nltk
from groq import Groq

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ── Text Extraction ──────────────────────────────────────────────────────────
def extract_text(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()
    elif ext == 'txt':
        raw = uploaded_file.read()
        return (raw.decode('utf-8') if isinstance(raw, bytes) else raw).strip()
    else:
        st.error("Upload a PDF or TXT file.")
        return ""

# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text, max_words=300, overlap=2):
    sentences = sent_tokenize(text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        current.append(sent)
        count += len(sent.split())
        if count >= max_words:
            chunks.append(" ".join(current))
            current = current[-overlap:]
            count = sum(len(s.split()) for s in current)
    if current:
        chunks.append(" ".join(current))
    return chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
def get_top_chunks(query, chunks, embeddings, top_k=6):
    from sentence_transformers import util
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0].cpu().numpy()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx], float(scores[top_idx[0]])

# ── Groq Answer ───────────────────────────────────────────────────────────────
def ask_groq(question, context_chunks, api_key):
    context = "\n\n".join(context_chunks)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using ONLY "
                    "the document context provided. If the answer is not in the context, "
                    "say 'This information is not found in the document.' "
                    "Be concise and direct."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.2,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="centered")
    st.title("📄 AskMyDocs")
    st.markdown("Upload a document and ask questions based on its content.")

    # API Key
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.sidebar.text_input("Groq API Key", type="password")
    if not api_key:
        st.warning("Enter your Groq API key in the sidebar to continue.")
        return

    # File Upload
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 15:
            st.error("File too large. Maximum 15MB.")
            return

        if "doc_name" not in st.session_state or st.session_state.doc_name != uploaded_file.name:
            with st.spinner("Processing document..."):
                text = extract_text(uploaded_file)
                if not text:
                    st.error("No text could be extracted from this file.")
                    return
                chunks = chunk_text(text)
                embeddings = model.encode(chunks, convert_to_tensor=True)
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.first_page = text[:1000]
                st.session_state.messages = []
            st.success(f"✅ Document ready — {len(chunks)} chunks indexed.")

        # Chat
        st.markdown("---")
        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask a question about your document...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    top_chunks, score = get_top_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.embeddings
                    )
                    # Always include first-page context for metadata questions
                    meta_q = any(w in question.lower() for w in ["author","who wrote","publisher","published","title","when","date"])
                    if meta_q:
                        top_chunks = [st.session_state.first_page] + top_chunks[:4]

                    answer = ask_groq(question, top_chunks, api_key)
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
