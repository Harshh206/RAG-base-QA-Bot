from src.ingestion.loader import load_documents
from src.ingestion.chunker import split_documents


def test_chunking():
    # Load documents
    docs = load_documents()

    # Ensure docs are loaded
    assert docs is not None
    assert len(docs) > 0, "No documents loaded"

    # Split documents
    chunks = split_documents(docs)

    # Ensure chunks are created
    assert chunks is not None
    assert len(chunks) > 0, "No chunks created"

    # Validate chunk structure
    assert hasattr(chunks[0], "page_content")
    assert hasattr(chunks[0], "metadata")

    # Debug output (optional)
    print(f"\n📄 Documents Loaded: {len(docs)}")
    print(f"✂️ Chunks Created: {len(chunks)}")

    print("\n🔹 First Chunk Content:")
    print(chunks[0].page_content[:200])

    print("\n🔹 First Chunk Metadata:")
    print(chunks[0].metadata)