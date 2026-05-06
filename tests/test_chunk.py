from src.ingestion.loader import load_documents
from src.ingestion.chunker import split_documents

docs = load_documents()
chunks = split_documents(docs)

print(len(docs), "→", len(chunks))
print(chunks[0].page_content[:200])
print(chunks[0].metadata)