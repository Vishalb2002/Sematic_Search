# Semantic_Search
# Sematic_Search
# Sematic_Search


# Semantic Search / Intelligent Q&A System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based Intelligent Question Answering System over a custom dataset of research papers.

Instead of performing keyword matching, the system converts documents into semantic embeddings using Sentence Transformers (SBERT), stores them in ChromaDB, retrieves the most relevant document chunks, re-ranks them using a CrossEncoder, and finally generates a natural language answer using a local Large Language Model (Llama 3.2 running through Ollama).

---

# Features

- PDF document ingestion
- Automatic text extraction
- Document chunking
- SBERT embedding generation
- ChromaDB vector database
- Semantic search
- CrossEncoder re-ranking
- Prompt Engineering
- Local LLM using Ollama
- Console-based Q&A interface

---

# Architecture

```
                 User Question
                       │
                       ▼
              SBERT Embedding
                       │
                       ▼
                ChromaDB Search
                       │
                Top 20 Chunks
                       │
                       ▼
            CrossEncoder ReRanker
                       │
                 Best 5 Chunks
                       │
                       ▼
                Prompt Builder
                       │
                       ▼
             Ollama (Llama 3.2)
                       │
                       ▼
                Final Answer
```

---

# Folder Structure

```
project/

│
├── data/
│   ├── raw/
│   ├── processed/
│   └── chunks/
│
├── vector_db/
│
├── src/
│
│   ├── loaders/
│   ├── preprocessing/
│   ├── embeddings/
│   ├── database/
│   ├── retrieval/
│   ├── llm/
│   ├── evaluation/
│   ├── config.py
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python
- Sentence Transformers (SBERT)
- Hugging Face
- ChromaDB
- CrossEncoder
- Ollama
- Llama 3.2
- PyMuPDF

---

# Workflow

1. Load PDF documents
2. Extract text
3. Split text into chunks
4. Generate SBERT embeddings
5. Store embeddings in ChromaDB
6. Retrieve top 20 relevant chunks
7. Re-rank using CrossEncoder
8. Build prompt
9. Generate final answer using Ollama

---

# Installation

Clone the repository

```bash
git clone <repository_url>
```

Create virtual environment

```bash
python -m venv venv
```

Install dependencies

```bash
pip install -r requirements.txt
```

Install Ollama

```bash
brew install ollama
```

Download Llama 3.2

```bash
ollama pull llama3.2:3b
```

Start Ollama

```bash
ollama serve
```

Run the application

```bash
python src/app.py
```

---

# Sample Output

```
Ask your question:

> What is Vision Transformer?

Answer:

Vision Transformer (ViT) is a deep learning architecture
based entirely on the Transformer model...
```

---

# Evaluation

The system was evaluated using qualitative analysis.

Ten representative queries were tested and the retrieved documents were manually inspected.

The retrieved chunks were found to be semantically relevant, demonstrating the effectiveness of embedding-based retrieval and CrossEncoder re-ranking.

---

# Future Enhancements

- Streamlit Web Interface
- Chat History
- Multi-turn Conversation
- Source Citation with Page Numbers
- Hybrid Search (BM25 + Dense Retrieval)
- Fine-tuned Embedding Models

---

# Author

Vishal Badhan