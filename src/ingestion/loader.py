from pathlib import Path
import pandas as pd

from config import DATA_PATH
from src.utils.excel_loader import load_excel_structured

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader, UnstructuredFileLoader


def load_documents():
    directory = Path(DATA_PATH)

    if not directory.exists() or not directory.is_dir():
        print(f"⚠️ Directory not found: {directory}")
        return []

    docs = []

    # PDFs
    for pdf_path in sorted(directory.glob("*.pdf")):
        docs.extend(PyMuPDFLoader(str(pdf_path)).load())

    # TXT
    for txt_path in sorted(directory.glob("*.txt")):
        docs.extend(TextLoader(str(txt_path), encoding="utf-8").load())

    # DOCX
    for docx_path in sorted(directory.glob("*.docx")):
        docs.extend(Docx2txtLoader(str(docx_path)).load())

    # Excel (structured)
    for excel_path in list(directory.glob("*.xlsx")) + list(directory.glob("*.xls")):
        docs.extend(load_excel_structured(excel_path))

    # Fallback (other file types)
    for file_path in directory.glob("*.*"):
        if file_path.suffix.lower() not in [".pdf", ".txt", ".docx", ".xlsx", ".xls"]:
            try:
                docs.extend(UnstructuredFileLoader(str(file_path)).load())
            except Exception as e:
                print(f"⚠️ Skipped {file_path.name}: {e}")

    return docs