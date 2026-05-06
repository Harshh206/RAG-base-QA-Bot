from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
                "If the answer is not in the context, say you don't know.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

