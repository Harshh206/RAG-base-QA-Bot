from pathlib import Path

# Model Names
MODEL_NAME = "llama3:8b" 
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw"
CHROMA_PATH = BASE_DIR / "db" / "chroma"

# Vector Store Logic
# This is the "folder" name inside your database
COLLECTION_NAME = "rag_docs_collection" 

# Chunking & Retrieval
CHUNK_SIZE = 600
CHUNK_OVERLAP = 85
RETRIEVER_K = 5