from src.vectorstore.chroma_store import get_vectorstore
from config import COLLECTION_NAME

def check_db_contents(limit: int = 10) -> None:
    vs = get_vectorstore(collection_name=COLLECTION_NAME)
    data = vs.get(limit=limit)

    ids = data.get("ids") or []
    if not ids:
        print("❌ The database is empty. You need to run your ingestion script first.")
    else:
        print(f"✅ Found {len(ids)} documents in the collection.\n")


def run_similarity_search(query: str = "Hello") -> None:
    vs = get_vectorstore(collection_name=COLLECTION_NAME)
    vs.add_texts(["Hello world", "LangChain with Ollama"])
    results = vs.similarity_search(query)
    print(results)


if __name__ == "__main__":
    check_db_contents()
    run_similarity_search()
