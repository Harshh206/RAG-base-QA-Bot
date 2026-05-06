from langchain_text_splitters import RecursiveCharacterTextSplitter 
from config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(documents):
    """
    Splits documents into smaller chunks based on CHUNK_SIZE and CHUNK_OVERLAP.
    Small documents (like structured Excel rows) are kept whole to preserve context.
    """
    if not documents:
        print("⚠️ No documents provided to split.")
        return []

    print(f"🔄 Splitting {len(documents)} documents...")

    # Initialize the splitter with values from config
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # Optional: helps track chunk location in source
        strip_whitespace=True
    )

    chunks = []

    for doc in documents:
        # ✅ Logic Check: Skip splitting for small structured docs (like Excel rows)
        # This prevents a single row of data from being fragmented.
        if len(doc.page_content) < CHUNK_SIZE:
            chunks.append(doc)
        else:
            # splitter.split_documents returns a list of Document objects 
            # while automatically preserving the metadata (source, type, etc.)
            split_docs = splitter.split_documents([doc])
            chunks.extend(split_docs)

    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents.")
    
    return chunks