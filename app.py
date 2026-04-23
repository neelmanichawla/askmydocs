import streamlit as st
import fitz  # PyMuPDF — installed as 'pymupdf' in requirements.txt
import re
import numpy as np
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# Cache models to avoid reloading
@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return retrieval_model, reranker

# Load models once
retrieval_model, reranker = load_models()

# Financial terms to filter out
FINANCIAL_TERMS = {'BOND', 'EQUITY', 'COMPANY', 'LLC', 'LTD', 'INC', 'CORP'}

# -----------------------------
# QUERY EXPANSION
# -----------------------------
def expand_query(question):
    """Expand query with synonyms for common question patterns"""
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
# DYNAMIC THRESHOLD
# -----------------------------
def dynamic_threshold(similarities, base=0.1):
    """Calculate dynamic threshold based on similarity distribution"""
    if len(similarities) == 0:
        return base
    mean = np.mean(similarities)
    std = np.std(similarities)
    # Use mean - 0.5*std so good matches (above average) always pass
    return max(base, mean - 0.5 * std)

# -----------------------------
# SCORE FILTERING
# -----------------------------
def filter_by_score(chunks, scores, min_score=0.2):
    """Filter chunks by minimum score threshold"""
    if isinstance(scores, (int, float)):
        return [c for c, s in zip(chunks, [scores]*len(chunks)) if s >= min_score]
    return [c for c, s in zip(chunks, scores) if s >= min_score]

# -----------------------------
# TEXT EXTRACTION
# -----------------------------
def extract_text(uploaded_file):
    """Extract text from PDF or TXT files"""
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == 'pdf':
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    elif file_type == 'txt':
        try:
            # Handle both file uploads and StringIO objects
            content = uploaded_file.read()
            if hasattr(content, 'decode'):
                return content.decode('utf-8').strip()
            else:
                return content.strip()
        except Exception as e:
            st.error(f"Error reading text file: {e}")
            return ""

    else:
        st.error("Unsupported file type. Please upload PDF or TXT.")
        return ""

# -----------------------------
# SEMANTIC CHUNKING
# -----------------------------
def chunk_text_semantic(text, max_words=400, overlap_sentences=2):
    """Chunk text by word count with sentence overlap"""
    sentences = sent_tokenize(text)
    chunks, current, count = [], [], 0

    for sent in sentences:
        current.append(sent)
        count += len(sent.split())
        if count >= max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]
            count = sum(len(s.split()) for s in current)

    if current:
        chunks.append(" ".join(current))
    return chunks

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    """Clean and normalize text"""
    # Fixed: proper regex escaping
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# SEMANTIC INDEX BUILDING
# -----------------------------
def build_semantic_index(chunks):
    """Build semantic index using sentence transformers"""
    return retrieval_model.encode(chunks, convert_to_tensor=True)

# -----------------------------
# SEMANTIC RETRIEVAL
# -----------------------------
def get_relevant_chunks_semantic(query, chunks, chunk_embeddings, top_k=5):
    """Find most relevant chunks using semantic similarity"""
    if not chunks:
        return [], 0.0, []

    from sentence_transformers import util
    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, chunk_embeddings)

    # Convert tensor to numpy for indexing
    scores_np = scores.cpu().numpy()[0] if hasattr(scores, 'cpu') else scores.numpy()[0]
    top_indices = np.argsort(scores_np)[::-1][:min(top_k, len(chunks))]

    relevant_chunks = [chunks[i] for i in top_indices]
    best_score = float(np.max(scores_np)) if len(scores_np) > 0 else 0.0

    return relevant_chunks, best_score, scores_np

# -----------------------------
# RERANKING
# -----------------------------
def rerank_chunks(query, chunks, top_k=5):
    """Rerank chunks using cross-encoder"""
    if not chunks:
        return chunks, []

    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), reverse=True)
    top_ranked = ranked[:top_k]
    return [chunk for _, chunk in top_ranked], [score for score, _ in top_ranked]

# -----------------------------
# INFORMATION EXTRACTION
# -----------------------------
def _extract_author(context):
    """Extract author information from context"""
    # Look for author declaration patterns
    author_patterns = [
        r'(?:author|written by|created by|prepared by)\s*:?\s*([^.]{5,100}?)(?=\n|[.;]|$)',
        r'by\s+([A-Z][a-zA-Z\s\.]{10,80}?)(?=\n|[.;]|$)',
        r'([A-Z][a-zA-Z\s\.]{15,100})\s+(?:wrote|authored|prepared)\s+(?:this|report|document)',
        r'(?:Report|Document|Paper)\s+(?:prepared|authored|written)\s+(?:by)?\s*:?\s*([^.]{5,100})'
    ]

    for pattern in author_patterns:
        match = re.search(pattern, context, re.IGNORECASE | re.MULTILINE)
        if match:
            author_candidate = match.group(1).strip()
            # Basic validation - should look like a name, not a company/financial term
            if (len(author_candidate) > 5 and
                not any(term in author_candidate.upper() for term in FINANCIAL_TERMS)):
                # Should have at least a first and last name
                words = [w for w in author_candidate.replace('.', ' ').split() if w]
                if len(words) >= 2:
                    clean_author = ' '.join(words)
                    if len(clean_author) > 5:
                        return clean_author
    return None

def _extract_date(context):
    """Extract date information from context"""
    # Look for date patterns
    date_patterns = [
        r'(?:dated|date)\s*:?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'([A-Za-z]+\s+\d{1,2},?\s+\d{4})'
    ]

    for pattern in date_patterns:
        match = re.search(pattern, context, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

# -----------------------------
# ANSWER GENERATION
# -----------------------------
def generate_answer(question, context_chunks):
    """Generate answer using semantic processing"""
    if not context_chunks:
        return "NOT FOUND IN DOCUMENT"

    # Join context for extraction
    full_context = " ".join(context_chunks)
    question_lower = question.lower()

    # Special handling for specific question types
    if any(word in question_lower for word in ['author', 'who wrote', 'created by', 'authored']):
        author = _extract_author(full_context)
        if author:
            return author

    elif any(word in question_lower for word in ['date', 'when', 'year']):
        date = _extract_date(full_context)
        if date:
            return date

    # For general questions, find the most relevant sentence
    best_answer = ""
    best_score = 0

    # Combine all context into sentences
    all_text = " ".join(context_chunks)
    sentences = sent_tokenize(all_text)

    # Score sentences by relevance to question
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if len(sentence_clean) > 15 and len(sentence_clean) < 300:
            # Count word overlaps
            sentence_words = set(sentence_clean.lower().split())
            question_words = set(question_lower.split())
            overlap_count = len(question_words.intersection(sentence_words))

            # Bonus for context words
            context_bonus = sum(1 for word in ['author', 'written', 'created', 'prepared', 'by', 'date']
                              if word in sentence_clean.lower()) * 2

            score = overlap_count + context_bonus

            if score > best_score:
                # Avoid financial terms unless strongly relevant
                if not any(term in sentence_clean.upper() for term in FINANCIAL_TERMS) or score > 5:
                    best_score = score
                    best_answer = sentence_clean

    # Return best answer if sufficiently relevant
    if best_answer and best_score >= 1:
        return best_answer

    return "NOT FOUND IN DOCUMENT"

# -----------------------------
# STREAMLIT UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="AskMyDocs - Improved RAG",
        page_icon="📄",
        layout="centered"
    )

    st.title("📄 AskMyDocs - Local RAG")
    st.markdown("Upload a document and ask questions based **only** on its content")

    # Sidebar for Settings
    with st.sidebar:
        st.header("Settings")
        st.info("✅ Local Processing Only - No API Keys Required!")
        st.markdown("---")
        st.markdown("### About AskMyDocs")
        st.markdown("This app processes documents entirely locally:")
        st.markdown("- Text extraction with PyMuPDF")
        st.markdown("- Semantic search with sentence transformers")
        st.markdown("- Local answer generation")
        st.markdown("- No external API dependencies")

    # Initialize session state
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "chunk_embeddings" not in st.session_state:  # New semantic embeddings
        st.session_state.chunk_embeddings = None
    if "last_file_name" not in st.session_state:
        st.session_state.last_file_name = ""

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Maximum file size: 15MB"
    )

    if uploaded_file:
        # Reset state if new file uploaded
        if uploaded_file.name != st.session_state.last_file_name:
            st.session_state.processed = False
            st.session_state.chunks = []
            st.session_state.chunk_embeddings = None
            st.session_state.last_file_name = uploaded_file.name

        # Check file size (approx 15MB limit)
        if uploaded_file.size > 15 * 1024 * 1024:
            st.error("File too large. Maximum size: 15MB")
            st.stop()

        # Process document
        if not st.session_state.processed or st.button("🔄 Reprocess Document"):
            # Reset file pointer to avoid stream exhaustion
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

    # Question section
    if st.session_state.processed:
        st.divider()
        st.header("❓ Ask a Question")

        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about this document?",
            help="The answer will be based only on the uploaded document content"
        )

        if question:
            # Expand query for better retrieval
            expanded_question = expand_query(question)

            with st.spinner("Searching for relevant information..."):
                # For metadata questions, bypass retrieval and use stored first-page text
                metadata_keywords = ['author', 'who wrote', 'created by', 'authored', 'publisher', 'published by', 'who made', 'who created']
                is_metadata_question = any(kw in question.lower() for kw in metadata_keywords)

                if is_metadata_question and st.session_state.get('doc_metadata'):
                    metadata = st.session_state['doc_metadata']
                    st.success("### Answer:")
                    st.write(f"Based on the document title page:\n\n{metadata[:500].strip()}")
                    st.stop()

                relevant_chunks, similarity_score, all_scores = get_relevant_chunks_semantic(
                    expanded_question,
                    st.session_state.chunks,
                    st.session_state.chunk_embeddings,
                    top_k=8
                )

                # Rerank chunks
                reranked_chunks, reranked_scores = rerank_chunks(expanded_question, relevant_chunks, top_k=5)

                # Filter by score
                filtered_chunks = filter_by_score(reranked_chunks, reranked_scores)

                st.info(f"Best similarity score: {similarity_score:.3f}")

                # Use dynamic threshold with all scores
                if similarity_score < dynamic_threshold(all_scores):
                    st.warning("NOT FOUND IN DOCUMENT")
                    st.info("The question doesn't match any content in the document.")
                else:
                    with st.spinner("Generating answer..."):
                        answer = generate_answer(question, filtered_chunks)

                        st.success("### Answer:")
                        st.write(answer)

                        # Debug info
                        if not answer or answer.strip() == "":
                            st.warning("DEBUG: Empty answer generated")
                        elif answer == "NOT FOUND IN DOCUMENT":
                            st.info("DEBUG: Answer filtered as not found in document")

                        # Show context with debug info
                        with st.expander("🔍 Debug: Retrieved Chunks"):
                            st.write(f"Retrieved {len(filtered_chunks)} chunks with best similarity: {similarity_score:.3f}")
                            for i, chunk in enumerate(filtered_chunks):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                                st.divider()

    # Instructions
    elif not uploaded_file:
        st.info("👆 Upload a PDF or TXT file to get started")

if __name__ == "__main__":
    main()