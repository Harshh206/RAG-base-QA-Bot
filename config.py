import os

# Ollama
OLLAMA_MODEL = "llama3:8b" # make sure you pulled this: ollama pull llama3

# Embeddings
EMBEDDING_MODEL = "qwen3-embedding:0.6b "

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/raw")
CHROMA_PATH = os.path.join(BASE_DIR, "db/chroma")

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100