from src.ingestion.loader import load_documents

def main():
    docs = load_documents()

    print(f"\n✅ Total documents loaded: {len(docs)}\n")

    if not docs:
        print("⚠️ No documents found. Check DATA_PATH.")
        return

    # Show first few documents
    for i, doc in enumerate(docs[:5]):
        print(f"\n--- Document {i+1} ---")
        print("Content preview:")
        print(doc.page_content[:300])  # first 300 chars
        print("\nMetadata:")
        print(doc.metadata)


if __name__ == "__main__":
    main()