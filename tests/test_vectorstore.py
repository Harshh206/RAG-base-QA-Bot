from src.vectorstore.chroma_store import get_vectorstore

vs = get_vectorstore()

vs.add_texts(["Hello world", "LangChain with Ollama"])

results = vs.similarity_search("Hello")

print(results)