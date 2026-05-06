from __future__ import annotations

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.llm.ollama_client import get_chat_model
from src.prompts.qa_prompt import build_qa_prompt
from src.vectorstore.chroma_store import get_vectorstore


def _format_docs(docs) -> str:
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)


def build_qa_chain(
    *,
    persist_dir: Path,
    collection_name: str,
    k: int,
    ollama_base_url: str,
    chat_model: str,
):
    vs = get_vectorstore(persist_dir=persist_dir, collection_name=collection_name)
    retriever = vs.as_retriever(search_kwargs={"k": k})

    prompt = build_qa_prompt()
    llm = get_chat_model(model=chat_model, base_url=ollama_base_url)

    chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(_format_docs),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.with_config(run_name="qa_chain").map(lambda answer: {"answer": answer})

