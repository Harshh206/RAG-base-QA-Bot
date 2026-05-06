from langchain_chroma import Chroma
from config import CHROMA_PATH
from src.embeddings.embedding_model import get_embeddings

def get_vectorstore(collection_name="document_collection"):
    """
    Initializes or loads a Chroma vectorstore.
    
    Args:
        collection_name (str): The name of the collection to store/retrieve chunks.
                              Defaults to "document_collection".
    Returns:
        Chroma: An instance of the Chroma vectorstore.
    """
    embeddings = get_embeddings()
    
    # We maintain the persist_directory from your config and the 
    # collection_name capability for better organization.
    return Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=collection_name,
        embedding_function=embeddings
    )