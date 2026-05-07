from src.llm.ollama_client import get_llm   # replace with your file name

def test_llm():
    llm = get_llm()
    
    try:
        response = llm.invoke("Say Hello .")
        print("Model response:")
        print(response)
    except Exception as e:
        print("Error occurred:")
        print(e)

if __name__ == "__main__":
    test_llm()