from langchain_classic.chains import RetrievalQA
from src.llm.ollama_client import get_llm
from src.vectorstore.chroma_store import get_vectorstore
from src.prompts.qa_prompt import qa_prompt

from config import (
    MODEL_NAME, 
    COLLECTION_NAME, 
    RETRIEVER_K)

def get_qa_chain():
    print("🔄 Initializing the QA Chain...")

    # 1.Initialize the LLM
    llm = get_llm()
    if not llm:
        print("⚠️ Error: Failed to initialize the LLM.")
        return None

    # 3. Load the VectorStore
    vectorstore = get_vectorstore(
        
        collection_name=COLLECTION_NAME
    )
    
    if not vectorstore:
        print("⚠️ Error: Failed to load the vector store.")
        return None

    # 3. Configure the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    try:
        # 4. Build the Final Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_prompt()},
            return_source_documents=True 
        )
        print(f"✅ QA Chain successfully built (Model: {MODEL_NAME}, Top K: {RETRIEVER_K}).")
        return qa_chain
        
    except Exception as e:
        print(f"⚠️ Failed to build QA Chain: {e}")
        return None