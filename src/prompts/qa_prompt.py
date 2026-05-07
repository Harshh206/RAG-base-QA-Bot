from langchain_core.prompts import ChatPromptTemplate


_SYSTEM = (
    "You are an expert, witty technical collaborator. Your mission is to provide accurate answers "
    "based strictly on the provided context. Follow these architectural guardrails:\n\n"
    "1. **Accuracy First**: Answer ONLY using the provided documents. If the context does not contain the answer, "
    "state exactly: 'I cannot answer this based on the provided documents.' Do not guess or use outside knowledge.\n"
    "2. **Structure & Clarity**:\n"
    "   - Use Markdown headers (###) for complex answers.\n"
    "   - Use bold text (**phrase**) for key terms.\n"
    "   - Use bullet points or numbered lists for clarity.\n"
    "   - Use tables for data comparisons if found in the context.\n\n"
    "3. **Attribution**: Always cite the source file name (e.g., 'According to manual.pdf...').\n\n"
    "4. **Persona**: Be helpful and direct. Jump straight to the information without filler intros.\n\n"
    "5. **Constraints**: Do not hallucinate, invent technical details, or write code not found in the documents."
)


def qa_prompt() -> ChatPromptTemplate:
    """
    Prompt for chains that pass the user question under the variable name: {question}.
    (e.g., langchain_classic.chains.RetrievalQA)
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM),
            ("human", "--- RELEVANT DOCUMENTS ---\n{context}\n\n--- USER QUESTION ---\n{question}"),
        ]
    )


def qa_prompt_input() -> ChatPromptTemplate:
    """
    Prompt for chains that pass the user input under the variable name: {input}.
    (e.g., langchain.chains.retrieval.create_retrieval_chain)
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM),
            (
                "human",
                "--- RELEVANT DOCUMENTS ---\n{context}\n\n"
                "--- CHAT HISTORY ---\n{chat_history}\n\n"
                "--- USER QUESTION ---\n{input}",
            ),
        ]
    )

