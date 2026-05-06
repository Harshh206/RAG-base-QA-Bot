from langchain_ollama import ChatOllama
from config import MODEL_NAME

def get_llm():
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.2,       # Strict adherence to facts
        num_ctx=4096,          # Accommodates multiple retrieved chunks
        top_p=0.9,             # Focuses on the most likely answers
        repeat_penalty=1.1,    # Prevents repetitive loops
    )
    return llm