# AskMyDocs - Local RAG Document Q&A

A powerful document question-answering system using Retrieval-Augmented Generation (RAG) with local processing and Ollama integration.

## Features

- 📄 **Document Processing**: Extract text from PDF and TXT files
- 🔍 **Smart Chunking**: Sentence-aware text segmentation with overlap
- 🎯 **TF-IDF Search**: Semantic search using scikit-learn
- 🤖 **Ollama Integration**: Multiple LLM model fallback (DeepSeek, MiniMax, GLM)
- 🚫 **Anti-Hallucination**: Strict prompting to prevent made-up answers
- 🌐 **Streamlit UI**: Clean, responsive web interface

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/askmydocs.git
   cd askmydocs
   ```

2. **Set up Miniconda environment**:
   ```bash
   conda create -n askmydocs python=3.10
   conda activate askmydocs
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama** (if not already installed):
   ```bash
   # Follow instructions at https://ollama.ai/
   ```

## Usage

1. **Start the application**:
   ```bash
   conda activate askmydocs
   streamlit run app.py
   ```

2. **Open your browser** to the provided URL (typically http://localhost:8501)

3. **Upload a document** (PDF or TXT)

4. **Ask questions** about the document content

## Project Structure

```
askmydocs/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── test_document.txt  # Sample test document
└── Trends_Artificial_Intelligence.pdf  # Sample PDF
```

## API Reference

### Main Functions

- `extract_text(uploaded_file)`: Extracts text from PDF or TXT files
- `chunk_text(text, chunk_size=500, overlap=100)`: Splits text into chunks
- `get_relevant_chunks_tfidf(query, chunks, vectorizer, chunk_vectors, top_k=8)`: Finds relevant chunks using TF-IDF
- `generate_answer(question, context_chunks)`: Generates answers using Ollama with fallback

### Configuration

- **Chunk Size**: 500 characters (adjustable)
- **Overlap**: 100 characters (adjustable)
- **Similarity Threshold**: 0.15
- **Models**: deepseek-v3.1:671b-cloud, minimax-m2:cloud, glm-4.6:cloud

## Deployment

### Render Deployment

1. **Connect your GitHub repository** to Render
2. **Create a Web Service** with the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Environment**: Python 3.10

### Environment Variables

No environment variables required for basic functionality.

## Limitations

- Maximum file size: ~15MB
- PDF text extraction depends on PyMuPDF quality
- TF-IDF search may not handle very long documents optimally
- Requires Ollama with cloud model access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.