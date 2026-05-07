import shutil
import time
from pathlib import Path

import gradio as gr
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config import CHROMA_PATH, COLLECTION_NAME, DATA_PATH, MODEL_NAME, RETRIEVER_K
from src.ingestion.chunker import split_documents
from src.ingestion.loader import load_documents
from src.llm.ollama_client import get_llm
from src.prompts.qa_prompt import qa_prompt_input
from src.vectorstore.chroma_store import get_vectorstore
from src.ingestion.upload_ingest import ingest_uploaded_files


CHAT_LOG_PATH = Path(__file__).resolve().parent / "chat_logs" / "gradio_chat.jsonl"


def _format_sources(source_documents):
    if not source_documents:
        return ""

    seen = set()
    items = []
    for doc in source_documents:
        source = None
        try:
            source = (doc.metadata or {}).get("source")
        except Exception:
            source = None
        if not source:
            continue
        
        # Using Path(source).name instead of os.path.basename
        name = Path(str(source)).name
        if name not in seen:
            seen.add(name)
            items.append(f"- {name}")

    return "\n".join(items)


def _history_to_text(history, max_turns=15):
    if not history:
        return ""

    lines = []

    for msg in history[-max_turns:]:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()

        if not content:
            continue

        if role == "user":
            lines.append(f"User: {content}")

        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    return "\n".join(lines).strip()


def _append_chat_log(user_text, assistant_text):
    try:
        CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not CHAT_LOG_PATH.exists():
            CHAT_LOG_PATH.write_text("", encoding="utf-8")
        
        with CHAT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f'{{"ts":"{ts}","user":{user_text!r},"assistant":{assistant_text!r}}}\n')
    except Exception:
        # Logging should never break the chat.
        pass


def build_rag_chain(chat_history_text):
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
    
    return (
        {
            "context": retriever,
            "input": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda _: chat_history_text),
        }
        | doc_chain
    )


def ingest_uploaded(files):
    return ingest_uploaded_files(
        files=files,
        collection_name=COLLECTION_NAME,
        split_documents=split_documents,
        get_vectorstore=get_vectorstore,
        log=lambda m: print(m, flush=True),
    )


def ingest():
    t0 = time.perf_counter()
    print("\n[ingest] Starting ingestion...", flush=True)

    db_path = Path(CHROMA_PATH)

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

def reset_database():
    """Safely reset the database using Chroma's native deletion instead of shutil."""
    try:
        vs = get_vectorstore(collection_name=COLLECTION_NAME)
        vs.delete_collection()
        time.sleep(0.5) # Give Windows a moment to catch its breath
        return f"✅ Database collection '{COLLECTION_NAME}' reset successfully."

    except Exception as e:
        return f"❌ Failed to reset database: {e}"

def answer(question, history):
    question = (question or "").strip()
    if not question:
        return (history or []), "Please enter a question."

    t0 = time.perf_counter()
    print("\n[qa] Question:", question, flush=True)

    # --- X-RAY DEBUGGER: Check what the database is actually finding ---
    try:
        temp_vs = get_vectorstore(collection_name=COLLECTION_NAME)
        found_docs = temp_vs.similarity_search(question, k=3)
        print(f"\n[debug] 🔍 The Database found {len(found_docs)} chunks to give the LLM:", flush=True)
        for i, d in enumerate(found_docs):
            source = d.metadata.get('source', 'unknown')
            preview = d.page_content.replace("\n", " ")[:80]
            print(f"  {i+1}. Source: {Path(source).name} | Text: {preview}...", flush=True)
        print("-" * 50, flush=True)
    except Exception as e:
        print(f"[debug] ❌ Search failed: {e}", flush=True)
    # -------------------------------------------------------------------

    history_text = _history_to_text(history)
    chain = build_rag_chain(chat_history_text=history_text)
    t_chain = time.perf_counter()
    
    answer_text = chain.invoke(question)
    t_done = time.perf_counter()

    print(f"[qa] Timings: build_chain={(t_chain - t0):.2f}s, invoke={(t_done - t_chain):.2f}s, total={(t_done - t0):.2f}s", flush=True)
    preview = (answer_text[:400] + "…") if len(answer_text) > 400 else answer_text
    print("[qa] Answer preview:\n" + preview, flush=True)

    new_history = list(history or [])

    new_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    new_history.append(
        {
            "role": "assistant",
            "content": answer_text,
        }
    )

    _append_chat_log(question, answer_text)

    return new_history, new_history



_CSS = """
/* ChatGPT-ish layout */
.container { max-width: 1200px !important; }
#sidebar { background: #0b1220; border-radius: 12px; padding: 14px; }
#sidebar h3, #sidebar label, #sidebar p { color: #e6e8ee !important; }
#main { border-radius: 12px; }
#chatbot { height: 72vh; }
"""

with gr.Blocks() as demo:
    state = gr.State([])

    with gr.Row(equal_height=True):
        with gr.Column(scale=3, min_width=320, elem_id="sidebar"):
            gr.Markdown("### RAG QA Assistant")
            gr.Markdown(
                f"**Model**: `{MODEL_NAME}`  \n"
                f"**DB**: `{CHROMA_PATH}`  \n"
                f"**Collection**: `{COLLECTION_NAME}`  \n"
            )

            reset_btn = gr.Button("Reset DB", variant="stop")
            reset_out = gr.Textbox(label="Reset Status", lines=2)
            
            gr.Markdown("### Upload & ingest")
            uploads = gr.File(label="Upload documents", file_count="multiple", type="filepath")
            ingest_upload_btn = gr.Button("Ingest uploaded files", variant="primary")
            ingest_upload_out = gr.Textbox(label="Status", lines=2)
            ingest_upload_btn.click(fn=ingest_uploaded, inputs=[uploads], outputs=ingest_upload_out)
            reset_btn.click(fn=reset_database, outputs=reset_out)
            


        with gr.Column(scale=7, elem_id="main"):
            chatbot = gr.Chatbot(
                label="",
                elem_id="chatbot"
            )

            with gr.Row():
                q = gr.Textbox(
                    label="",
                    placeholder="Message…",
                    lines=2,
                    autofocus=True,
                    scale=8,
                )
                ask_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)



            q.submit(answer,[q, state],[state, chatbot]).then(lambda: "",None,q)
            ask_btn.click(answer,[q, state],[state, chatbot]).then(lambda: "",None,q)
            clear_btn.click(lambda: ([], []), None, [state, chatbot])


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=gr.themes.Soft(),css=_CSS, show_error=True)