from src.embeddings.embedding_model import get_embeddings

#Testing Initialize embeddings
embeddings = get_embeddings()

# Test with sample text
text = "Hello world"
vector = embeddings.embed_query(text)

print("Embedding vector length:", len(vector))
print("First 5 values:", vector[:5])