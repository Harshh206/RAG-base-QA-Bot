from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL

def get_llm():
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.2
    )
    return llm