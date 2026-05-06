from src.ingestion.loader import load_documents
from src.ingestion.chunker import split_documents
from src.embeddings.embedding_model import get_embeddings
from langchain_chroma import Chroma
from config import CHROMA_PATH, COLLECTION_NAME

def main():
    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Creating embeddings...")
    embeddings = get_embeddings()

    print("Storing in ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=str(CHROMA_PATH), # <-- Make sure this is here!
        collection_name=COLLECTION_NAME
        )
    
    # db.persist()  <--- DELETE OR COMMENT OUT THIS LINE
    print("✅ Successfully saved to ChromaDB.")

if __name__ == "__main__":
    main()