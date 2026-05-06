from __future__ import annotations

import os
import shutil
import time
from typing import Any

import gradio as gr
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from config import CHROMA_PATH, COLLECTION_NAME, DATA_PATH, MODEL_NAME, RETRIEVER_K
from src.ingestion.chunker import split_documents
from src.ingestion.loader import load_documents
from src.llm.ollama_client import get_llm
from src.prompts.prompt import qa_prompt_input
from src.vectorstore.chroma_store import get_vectorstore


def _format_sources(source_documents: list[Any] | None) -> str:
    if not source_documents:
        return ""

    seen: set[str] = set()
    items: list[str] = []
    for doc in source_documents:
        source = None
        try:
            source = (doc.metadata or {}).get("source")
        except Exception:
            source = None
        if not source:
            continue
        name = os.path.basename(str(source))
        if name not in seen:
            seen.add(name)
            items.append(f"- {name}")

    return "\n".join(items)


def build_rag_chain():
    """
    Build a retrieval-augmented generation chain using:
    - Chroma (persisted at CHROMA_PATH)
    - Ollama chat model (MODEL_NAME via config)
    - Prompt defined in src/prompts/prompt.py
    """
    vs = get_vectorstore(collection_name=COLLECTION_NAME)
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = get_llm()
    prompt = qa_prompt_input()

    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever, doc_chain)


def ingest(reset_db: bool = False) -> str:
    t0 = time.perf_counter()
    print("\n[ingest] Starting ingestion...", flush=True)

    if reset_db:
        try:
            if os.path.exists(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
            print(f"[ingest] Reset DB folder: {CHROMA_PATH}", flush=True)
        except Exception as e:
            print(f"[ingest] Failed to reset DB: {e}", flush=True)
            return f"❌ Failed to reset DB at {CHROMA_PATH}: {e}"

    docs = load_documents()
    if not docs:
        print("[ingest] No documents found.", flush=True)
        return f"⚠️ No documents found in {DATA_PATH}"

    splits = split_documents(docs)
    if not splits:
        print("[ingest] No chunks created.", flush=True)
        return "⚠️ No chunks created."

    vs = get_vectorstore(collection_name=COLLECTION_NAME)
    vs.add_documents(splits)

    elapsed = time.perf_counter() - t0
    msg = f"✅ Ingested {len(docs)} docs → {len(splits)} chunks into '{COLLECTION_NAME}'"
    print(f"[ingest] Done in {elapsed:.2f}s. {msg}", flush=True)
    return msg


def answer(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return "Please enter a question."

    t0 = time.perf_counter()
    print("\n[qa] Question:", question, flush=True)
    chain = build_rag_chain()
    t_chain = time.perf_counter()
    result = chain.invoke({"input": question})
    t_done = time.perf_counter()

    answer_text = result.get("answer") or ""
    ctx = result.get("context")
    sources_text = _format_sources(ctx)

    ctx_count = len(ctx) if isinstance(ctx, list) else 0
    print(f"[qa] Retrieved docs: {ctx_count}", flush=True)
    if sources_text:
        print("[qa] Sources:\n" + sources_text, flush=True)
    print(f"[qa] Timings: build_chain={(t_chain - t0):.2f}s, invoke={(t_done - t_chain):.2f}s, total={(t_done - t0):.2f}s", flush=True)
    preview = (answer_text[:400] + "…") if len(answer_text) > 400 else answer_text
    print("[qa] Answer preview:\n" + preview, flush=True)

    if sources_text:
        return f"{answer_text}\n\nSources:\n{sources_text}"
    return answer_text


with gr.Blocks() as demo:
    gr.Markdown("# 🤖 RAG QA Assistant (LangChain + Ollama + Chroma)")
    gr.Markdown(
        f"- **Model**: `{MODEL_NAME}`\n"
        f"- **Chroma DB**: `{CHROMA_PATH}`\n"
        f"- **Collection**: `{COLLECTION_NAME}`\n"
        f"- **Data folder**: `{DATA_PATH}`"
    )

    with gr.Tab("Knowledge Base"):
        reset = gr.Checkbox(value=False, label="Reset DB before ingest (recommended if results look wrong)")
        ingest_btn = gr.Button("Build / Update Vector Database", variant="primary")
        ingest_out = gr.Textbox(label="Status", lines=2)
        ingest_btn.click(fn=ingest, inputs=reset, outputs=ingest_out)

    with gr.Tab("Ask"):
        q = gr.Textbox(
            label="Question",
            placeholder="Ask a question about your documents...",
            lines=2,
        )
        ask_btn = gr.Button("Submit", variant="primary")
        a = gr.Textbox(label="Answer", lines=12)
        ask_btn.click(fn=answer, inputs=q, outputs=a)
        q.submit(fn=answer, inputs=q, outputs=a)


if __name__ == "__main__":
    # Queue helps when Ollama responses take ~1-2 minutes.
    demo.queue()
    demo.launch(theme=gr.themes.Soft(), show_error=True)