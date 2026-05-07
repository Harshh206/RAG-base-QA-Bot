## 🤖  RAG QA Bot with LangChain and Local LLMs

A powerful **Retrieval-Augmented Generation (RAG)** based Question Answering Bot built using:

- LangChain
- Ollama (Local LLM)
- ChromaDB
- Gradio

This project allows users to upload documents (PDF, TXT, Excel, etc.), create embeddings, store them in a vector database, and ask context-aware questions using a local LLM.



### 🚀 Features

- ✅ Local LLM support using Ollama
- ✅ RAG-based Question Answering
- ✅ PDF ingestion and chunking
- ✅ Excel file formatting and ingestion
- ✅ ChromaDB vector storage
- ✅ HuggingFace embeddings
- ✅ Gradio web interface
- ✅ Modular project architecture
- ✅ Persistent vector database
- ✅ Unit testing support



### 📁 Project Structure

```text
RAG-QA-Bot-Using-LangChain/
│
├── app.py                      # Main Gradio app entry point
├── config.py                   # Central config (model names, paths, etc.)
├── requirements.txt            # Dependencies
├── .env                        # API keys (if any, keep secret)
├── .gitignore
│
├── src/                        # Core logic
│   │
│   ├── chains/                 # LangChain pipelines
│   │   └── qa_chain.py
│   │
│   ├── embeddings/             # Embedding models
│   │   └── embedding_model.py
│   │
│   ├── ingestion/              # Data loading & chunking
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── excel_loader.py     # Format Excel for ingestion
│   │   └── upload_ingest.py    # Upload and ingest files
│   │
│   ├── llm/                    # LLM setup (Ollama)
│   │   └── ollama_client.py
│   │
│   ├── prompts/                # Prompt templates
│   │   └── qa_prompt.py
│   │
│   ├── vectorstore/            # ChromaDB setup
│   │   └── chroma_store.py
│   │
│   └── utils/                  # Helpers
│       └── logger.py
│
├── data/                       # Input documents
│   ├── raw/                    
│   └── processed/
│
├── db/                         # ChromaDB persistent storage
│   └── chroma/
│
└── tests/                      # Unit tests
    ├── __init__.py
    ├── test_chunk.py
    ├── test_embedding.py
    ├── test_llm.py
    ├── test_loader.py
    ├── test_qa.py
    └── test_vectorstore.py
```
### ⚙️ Installation
#### 1️⃣ Clone Repository
```
git clone https://github.com/your-username/RAG-QA-Bot-Using-LangChain.git
cd RAG-QA-Bot-Using-LangChain
```
#### 2️⃣ Create Virtual Environment

``` 
Windows 
py --3.11 -m venv .venv   # Use python 3.11
.venv\Scripts\activate

Linux / Mac
python3.11 -m venv .venv
source .venv/bin/activate
```
#### 3️⃣ Install Dependencies
```
pip install -r requirements.tx
```
### 🦙Install Ollama 
[Download Ollama](https://ollama.com/)
>Pull a model:
```
ollama pull llama3:8b  # Chat Model
ollama pull qwen3-embedding:0.6b  # Embedding Model

```
### 📄 Add Documents
Place your documents inside
``data/raw/``. Supported formats: PDF, TXT, Excel, docx, etc.
### 🧠 Create Embeddings
Run ingestion script ``python scripts/ingest.py``. 
This will:
* Load documents
* Split into chunks
* Generate embeddings
* Store vectors in ChromaDB

### 💾 Where Embeddings Are Stored
Embeddings are stored locally inside ``db/chroma/``. This folder contains:
* Vector embeddings
* ChromaDB metadata
* SQLite database
* Search indexes

### ▶️ Run Application
```
python app.py
```
Gradio UI will start locally:
```
http://127.0.0.1:7860
```
### 🔄 RAG Workflow
```
Documents
   ↓
Loader
   ↓
Chunking
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Retriever
   ↓
Ollama LLM
   ↓
Answer
```
### 🧪 Running Tests
Run all tests:
```
pytest
```
Run individual test:
```
py -m tests.test_llm
py -m tests.test_loader
py -m tests.test_chunk
py -m tests.test_embedding
py -m tests.test_vectorstore
py -m tests.test_qa
```
---
