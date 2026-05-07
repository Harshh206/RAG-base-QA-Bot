from src.embeddings.embedding_model import get_embeddings


def test_embeddings():
    # Initialize embeddings model
    embeddings = get_embeddings()

    # Ensure model loaded
    assert embeddings is not None

    # Sample text
    text = "Hello world"

    # Generate embedding
    vector = embeddings.embed_query(text)

    # Validate embedding output
    assert vector is not None
    assert isinstance(vector, list)
    assert len(vector) > 0

    # Debug output
    print(f"\n✅ Embedding vector length: {len(vector)}")
    print(f"🔹 First 5 values: {vector[:5]}")