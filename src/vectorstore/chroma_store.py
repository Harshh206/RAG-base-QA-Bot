from langchain_chroma import Chroma
from config import CHROMA_PATH
from src.embeddings.embedding_model import get_embeddings

def get_vectorstore():
    embeddings = get_embeddings()
    
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )