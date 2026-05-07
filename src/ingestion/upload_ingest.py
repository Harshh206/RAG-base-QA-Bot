import time
import hashlib
from pathlib import Path

from src.ingestion.loader import load_single_document


def load_documents_from_paths(file_paths):
    """Load Documents from multiple uploaded file paths."""
    docs = []
    for path_str in file_paths:
        if path_str:
            print(f"[loader] Processing: {path_str}")
            docs.extend(load_single_document(path_str))
    return docs

def generate_deterministic_ids(splits):
    """Generates unique IDs based on content so re-uploading the same file doesn't duplicate DB entries."""
    ids = []
    for i, chunk in enumerate(splits):
        # Create a unique but repeatable hash based on the text and the source file
        content = chunk.page_content.encode("utf-8")
        content_hash = hashlib.md5(content).hexdigest()
        source = chunk.metadata.get("source", "unknown_source")
        
        # ID format: source_file_path-chunk_index-content_hash
        chunk_id = f"{source}-{i}-{content_hash}"
        ids.append(chunk_id)
    return ids

def ingest_uploaded_files(
    *,
    files,
    collection_name,
    split_documents,
    get_vectorstore,
    log=None,
):
    """Ingest multiple uploaded files into Chroma vectorstore, replacing old versions of the same files."""

    def _log(msg):
        if log:
            log(msg)
        else:
            print(msg)

    t0 = time.perf_counter()
    _log("\n[upload-ingest] Starting targeted upload ingestion...")

    # ---------------- VALIDATE FILES ----------------
    file_paths = [f for f in (files or []) if f]

    if not file_paths:
        return "⚠️ Please upload at least one file."

    _log(f"[upload-ingest] Uploaded files: {len(file_paths)}")

    # ---------------- LOAD DOCS ----------------
    docs = load_documents_from_paths(file_paths)

    if not docs:
        return "⚠️ No documents could be loaded."

    _log(f"[upload-ingest] Loaded docs: {len(docs)}")

    # ---------------- SPLIT DOCS ----------------
    splits = split_documents(docs)

    if not splits:
        return "⚠️ No chunks created."

    _log(f"[upload-ingest] Created chunks: {len(splits)}")

    # ---------------- VECTORSTORE TARGETED UPDATE ----------------
    try:
        vs = get_vectorstore(collection_name=collection_name)
        
        # 1. Identify unique sources from the current upload batch
        unique_sources = set(doc.metadata.get("source") for doc in docs if "source" in doc.metadata)
        
        # 2. Delete existing data for these specific files
        for source in unique_sources:
            try:
                # Access the native Chroma collection to delete by metadata
                vs._collection.delete(where={"source": source})
                _log(f"[upload-ingest] Removed old database entries for: {Path(source).name}")
            except Exception as e:
                # It's fine if it fails; it usually just means the file wasn't in the DB yet
                pass 

        # 3. Add the new data
        # Use deterministic IDs to prevent duplicates
        chunk_ids = generate_deterministic_ids(splits)
        vs.add_documents(
            splits,
            ids=chunk_ids,
        )

        # Persist if supported (Chroma v0.4+ handles this automatically, but good for backwards compat)
        if hasattr(vs, "persist"):
            vs.persist()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Failed during vectorstore ingestion: {e}"

    # ---------------- DONE ----------------
    elapsed = time.perf_counter() - t0

    msg = (
        f"✅ Targeted ingest successful\n"
        f"Files Updated: {len(unique_sources)}\n"
        f"New Chunks Added: {len(splits)}\n"
        f"Collection: {collection_name}\n"
        f"Time: {elapsed:.2f}s"
    )

    _log(f"[upload-ingest] Done in {elapsed:.2f}s")
    return msg