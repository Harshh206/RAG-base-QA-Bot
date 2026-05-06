from __future__ import annotations

import gradio as gr

from config import CONFIG
from src.chains.qa_chain import build_qa_chain
from src.ingestion.loader import load_documents_from_dir
from src.ingestion.chunker import chunk_documents
from src.utils.logger import get_logger
from src.vectorstore.chroma_store import get_vectorstore

logger = get_logger(__name__)


def ingest() -> str:
    raw_dir = CONFIG.data_raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents_from_dir(raw_dir)
    if not docs:
        return f"No documents found in: {raw_dir}"

    splits = chunk_documents(docs, chunk_size=CONFIG.chunk_size, chunk_overlap=CONFIG.chunk_overlap)
    vs = get_vectorstore(persist_dir=CONFIG.chroma_persist_dir, collection_name=CONFIG.collection_name)
    vs.add_documents(splits)
    return f"Ingested {len(docs)} document(s), created {len(splits)} chunks into collection '{CONFIG.collection_name}'."


def answer(question: str) -> str:
    chain = build_qa_chain(
        persist_dir=CONFIG.chroma_persist_dir,
        collection_name=CONFIG.collection_name,
        k=CONFIG.k,
        ollama_base_url=CONFIG.ollama_base_url,
        chat_model=CONFIG.chat_model,
    )
    return chain.invoke({"question": question})["answer"]


with gr.Blocks(title="RAG QA Bot") as demo:
    gr.Markdown("## RAG QA Bot (LangChain + Chroma + Ollama)")

    with gr.Row():
        ingest_btn = gr.Button("Ingest documents from data/raw")
        ingest_out = gr.Textbox(label="Ingestion status", interactive=False)
        ingest_btn.click(fn=ingest, outputs=ingest_out)

    q = gr.Textbox(label="Question", placeholder="Ask a question about your documents...")
    a = gr.Textbox(label="Answer", lines=8, interactive=False)
    ask_btn = gr.Button("Ask")
    ask_btn.click(fn=answer, inputs=q, outputs=a)


if __name__ == "__main__":
    logger.info("Starting Gradio app")
    demo.launch()

