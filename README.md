# 📄 AskMyDocs

Ask questions about any PDF or TXT document. Get answers grounded strictly in your document — no hallucination, no outside knowledge.

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://askmydocs.streamlit.app)

## ✨ Features

- Upload any **PDF or TXT** file (up to 15MB)
- Ask **any question** — factual, summary, explanation
- Answers grounded **only** in your document
- Fully **free** — no credit card needed
- Built on state-of-the-art retrieval pipeline

## 🧠 How It Works

```
Your Document
     ↓
Sentence-aware chunking (150 words, 3-sentence overlap)
     ↓
Hybrid Retrieval — BGE embeddings + BM25 via Reciprocal Rank Fusion
     ↓
Cross-encoder Reranking — picks most precise chunks
     ↓
Groq LLM (llama-3.1-8b-instant) — generates answer from chunks only
     ↓
Answer
```

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| UI | Streamlit |
| PDF extraction | PyMuPDF |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Keyword search | BM25 (rank_bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API — `llama-3.1-8b-instant` |

## ⚙️ Deploy Your Own (Free)

### 1. Fork this repo

### 2. Get a free Groq API key
Go to [console.groq.com](https://console.groq.com) — no credit card required.

### 3. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect this repo
3. In **Advanced settings → Secrets**, add:
```toml
GROQ_API_KEY = "your_key_here"
```
4. Click Deploy

### 4. Done — live in ~3 minutes ✅

## 💻 Run Locally

```bash
# Create environment
conda create -n askmydocs python=3.10 -y
conda activate askmydocs

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
mkdir .streamlit
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# Run
streamlit run app.py
```

## 📁 Project Structure

```
askmydocs/
├── app.py              ← Main application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

## ⚠️ Privacy Note

Documents are processed in memory only and never stored. Do not upload sensitive or confidential documents to the public deployment.

## 📜 License

MIT
