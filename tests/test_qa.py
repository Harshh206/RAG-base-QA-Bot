from src.chains.qa_chain import get_qa_chain

def test_system():
    # Initialize the chain
    chain = get_qa_chain()
    
    if not chain:
        print("❌ Chain failed to initialize.")
        return

    # Ask a question that SHOULD be in your documents
    query = "What is the main topic of the uploaded documents?"
    
    print(f"\n🔍 Testing Query: {query}")
    response = chain.invoke({"query": query})

    # 1. Check the Answer
    print("\n🤖 AI Response:")
    print(response["result"])

    # 2. Check the Sources (This confirms Retrieval is working)
    print("\n📄 Source Documents Used:")
    if response["source_documents"]:
        for i, doc in enumerate(response["source_documents"]):
            print(f"--- Source {i+1} ---")
            print(f"Content Snippet: {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")
    else:
        print("⚠️ No sources found! Check if your VectorStore is empty.")

if __name__ == "__main__":
    test_system()