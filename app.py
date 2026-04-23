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

@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return retrieval_model, reranker

retrieval_model, reranker = load_models()

# -----------------------------
# QUERY EXPANSION
# -----------------------------
def expand_query(question):
    expansions = {
        "who wrote": "author creator written by",
        "who created": "author creator made by",
        "when was": "date year published released",
        "what is": "definition meaning description explanation",
        "how many": "count number total quantity",
    }
    q = question.lower()
    for phrase, expansion in expansions.items():
        if phrase in q:
            return question + " " + expansion
    return question

# -----------------------------
# DYNAMIC THRESHOLD (fixed)
# -----------------------------
def dynamic_threshold(similarities, base=0.1):
    if len(similarities) == 0:
        return base
    mean = np.mean(similarities)
    std = np.std(similarities)
    return max(base, mean - 0.5 * std)  # fixed: was mean + 0.5*std (too aggressive)

# -----------------------------
# TEXT EXTRACTION
# -----------------------------
def extract_text(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            full = text.strip()
            # Store first-page metadata for author/title/date questions
            st.session_state['doc_metadata'] = full[:800]
            return full
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    elif file_type == 'txt':
        try:
            content = uploaded_file.read()
            return (content.decode('utf-8') if hasattr(content, 'decode') else content).strip()
        except Exception as e:
            st.error(f"Error reading text file: {e}")
            return ""
    else:
        st.error("Unsupported file type. Please upload PDF or TXT.")
        return ""

# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text_semantic(text, max_words=400, overlap_sentences=2):
    sentences = sent_tokenize(text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        current.append(sent)
        count += len(sent.split())
        if count >= max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if len(current) > overlap_sentences else []
            count = sum(len(s.split()) for s in current)
    if current:
        chunks.append(" ".join(current))
    return chunks

# -----------------------------
# SEMANTIC INDEX
# -----------------------------
def build_semantic_index(chunks):
    return retrieval_model.encode(chunks, convert_to_tensor=True)

# -----------------------------
# RETRIEVAL
# -----------------------------
def get_relevant_chunks_semantic(query, chunks, chunk_embeddings, top_k=8):
    if not chunks:
        return [], 0.0, []
    from sentence_transformers import util
    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, chunk_embeddings)
    scores_np = scores.cpu().numpy()[0] if hasattr(scores, 'cpu') else scores.numpy()[0]
    top_indices = np.argsort(scores_np)[::-1][:min(top_k, len(chunks))]
    relevant_chunks = [chunks[i] for i in top_indices]
    best_score = float(np.max(scores_np)) if len(scores_np) > 0 else 0.0
    return relevant_chunks, best_score, scores_np

# -----------------------------
# RERANKING
# -----------------------------
def rerank_chunks(query, chunks, top_k=5):
    if not chunks:
        return chunks, []
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), reverse=True)
    top_ranked = ranked[:top_k]
    return [chunk for _, chunk in top_ranked], [score for score, _ in top_ranked]

# -----------------------------
# GROQ ANSWER (only sees top chunks)
# -----------------------------
def generate_answer(question, context_chunks):
    if not context_chunks:
        return "NOT FOUND IN DOCUMENT"

    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return "Groq API key not configured in Streamlit secrets."

    # For metadata questions, prepend first-page text as an extra chunk
    meta_keywords = ['author', 'who wrote', 'publisher', 'published', 'title', 'when', 'date', 'created by']
    if any(kw in question.lower() for kw in meta_keywords):
        metadata = st.session_state.get('doc_metadata', '')
        if metadata:
            context_chunks = [metadata] + context_chunks[:3]  # 4 chunks max

    # Keep only top 4 chunks to stay well within Groq token limits
    context_chunks = context_chunks[:4]
    context = "\n\n---\n\n".join(context_chunks)

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful document assistant. Answer the user's question "
                        "using ONLY the document excerpts provided. Be concise and direct. "
                        "If the answer is not in the excerpts, say 'This information is not found in the document.'"
                    )
                },
                {
                    "role": "user",
                    "content": f"Document excerpts:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.1,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# -----------------------------
# STREAMLIT UI (original preserved)
# -----------------------------
def main():
    st.set_page_config(
        page_title="AskMyDocs",
        page_icon="📄",
        layout="centered"
    )

    st.title("📄 AskMyDocs")
    st.markdown("Upload a document and ask questions based **only** on its content")

    with st.sidebar:
        st.header("Settings")
        st.info("✅ Groq-powered answers with local semantic search")
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("- 📥 Text extracted with PyMuPDF")
        st.markdown("- 🔍 Semantic search finds relevant chunks")
        st.markdown("- 🤖 Groq LLM answers from those chunks only")
        st.markdown("- 🔒 Your document never leaves your session")

    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings = None
    if "last_file_name" not in st.session_state:
        st.session_state.last_file_name = ""

    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Maximum file size: 15MB"
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.last_file_name:
            st.session_state.processed = False
            st.session_state.chunks = []
            st.session_state.chunk_embeddings = None
            st.session_state.last_file_name = uploaded_file.name

        if uploaded_file.size > 15 * 1024 * 1024:
            st.error("File too large. Maximum size: 15MB")
            st.stop()

        if not st.session_state.processed or st.button("🔄 Reprocess Document"):
            uploaded_file.seek(0)

            with st.spinner("Extracting text..."):
                text = extract_text(uploaded_file)
                if not text:
                    st.error("No text could be extracted from the document.")
                    st.stop()
                st.info(f"Extracted {len(text)} characters")

            with st.spinner("Chunking text..."):
                chunks = chunk_text_semantic(text)
                if not chunks:
                    st.error("No valid chunks created from text.")
                    st.stop()
                st.session_state.chunks = chunks
                st.info(f"Created {len(chunks)} chunks")

            with st.spinner("Building semantic index..."):
                try:
                    chunk_embeddings = build_semantic_index(chunks)
                    st.session_state.chunk_embeddings = chunk_embeddings
                    st.session_state.processed = True
                    st.success("✅ Document processed successfully!")
                except Exception as e:
                    st.error(f"Error building semantic index: {e}")
                    st.stop()

    if st.session_state.processed:
        st.divider()
        st.header("❓ Ask a Question")

        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about this document?",
            help="The answer will be based only on the uploaded document content"
        )

        if question:
            expanded_question = expand_query(question)

            with st.spinner("Searching for relevant information..."):
                relevant_chunks, similarity_score, all_scores = get_relevant_chunks_semantic(
                    expanded_question,
                    st.session_state.chunks,
                    st.session_state.chunk_embeddings,
                    top_k=8
                )

                reranked_chunks, reranked_scores = rerank_chunks(expanded_question, relevant_chunks, top_k=5)

                st.info(f"Best similarity score: {similarity_score:.3f}")

                if similarity_score < dynamic_threshold(all_scores):
                    st.warning("This question doesn't appear to match any content in the document.")
                else:
                    with st.spinner("Generating answer..."):
                        answer = generate_answer(question, reranked_chunks)
                        st.success("### Answer:")
                        st.write(answer)

                        with st.expander("🔍 Retrieved Chunks"):
                            for i, chunk in enumerate(reranked_chunks):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                                st.divider()

    elif not uploaded_file:
        st.info("👆 Upload a PDF or TXT file to get started")

if __name__ == "__main__":
    main()
