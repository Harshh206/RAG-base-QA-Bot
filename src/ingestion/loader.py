from pathlib import Path
import pandas as pd

from config import DATA_PATH
from src.utils.excel_loader import load_excel_structured

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain_unstructured import UnstructuredLoader

def load_documents():
    directory = Path(DATA_PATH)

    if not directory.exists() or not directory.is_dir():
        print(f"⚠️ Directory not found: {directory}")
        return []

    docs = []
    
    # We use a set for O(1) lookups during the fallback check
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".xlsx", ".xls"}

    # Use iterdir() to process each file once efficiently
    for file_path in sorted(directory.iterdir()):
        if file_path.is_dir():
            continue
            
        ext = file_path.suffix.lower()

        try:
            # --- PDF ---
            if ext == ".pdf":
                for d in PyMuPDFLoader(str(file_path)).load():
                    docs.append(Document(
                        page_content=d.page_content,
                        metadata={"source": str(file_path), "type": "pdf"}
                    ))

            # --- TXT ---
            elif ext == ".txt":
                for d in TextLoader(str(file_path), encoding="utf-8").load():
                    docs.append(Document(
                        page_content=d.page_content,
                        metadata={"source": str(file_path), "type": "txt"}
                    ))

            # --- DOCX ---
            elif ext == ".docx":
                for d in Docx2txtLoader(str(file_path)).load():
                    docs.append(Document(
                        page_content=d.page_content,
                        metadata={"source": str(file_path), "type": "docx"}
                    ))

            # --- Excel ---
            elif ext in [".xlsx", ".xls"]:
                # Assuming load_excel_structured already returns Documents 
                # with the correct metadata format per your previous code
                docs.extend(load_excel_structured(file_path))

            # --- Fallback (Other) ---
            elif file_path.suffix: # Ensure it's actually a file with an extension
                try:
                    # Using UnstructuredLoader as requested in your imports
                    loader = UnstructuredLoader(str(file_path))
                    for d in loader.load():
                        docs.append(Document(
                            page_content=d.page_content,
                            metadata={"source": str(file_path), "type": "other"}
                        ))
                except Exception as e:
                    print(f"⚠️ Skipped {file_path.name}: {e}")

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

    return docs