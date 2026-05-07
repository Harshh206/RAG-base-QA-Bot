from pathlib import Path
from config import DATA_PATH

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from src.ingestion.excel_loader import load_excel_structured

def load_single_document(file_path) -> list[Document]:
    """Loads a single file and returns a list of LangChain Documents with consistent metadata."""
    path = Path(file_path)
    
    if not path.exists() or not path.is_file():
        return []

    ext = path.suffix.lower()
    docs = []

    try:
        # --- Core Loaders ---
        if ext == ".pdf":
            docs = PyMuPDFLoader(str(path)).load()
        elif ext == ".txt":
            docs = TextLoader(str(path), encoding="utf-8").load()
        elif ext == ".docx":
            docs = Docx2txtLoader(str(path)).load()
        elif ext in [".xlsx", ".xls"]:
            docs = load_excel_structured(str(path))
            
        # --- Fallback Loader ---
        else:
            # Lazy import to speed up processing for standard files
            from langchain_unstructured import UnstructuredLoader
            docs = UnstructuredLoader(str(path)).load()

        # --- Enforce Consistent Metadata ---
        doc_type = ext.lstrip('.') if ext else "other"
        
        for doc in docs:
            # Overwrite or set standard metadata so filtering works flawlessly later
            doc.metadata["source"] = str(path)
            doc.metadata["type"] = doc_type

    except Exception as e:
        print(f"❌ Error loading {path.name}: {e}")

    return docs

def load_documents():
    """Batch loads all supported documents from the configured DATA_PATH."""
    directory = Path(DATA_PATH)

    if not directory.exists() or not directory.is_dir():
        print(f"⚠️ Directory not found: {directory}")
        return []

    all_docs = []
    
    # Process all files in the directory
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file():
            all_docs.extend(load_single_document(file_path))

    return all_docs