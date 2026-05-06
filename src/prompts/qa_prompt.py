from langchain_core.prompts import ChatPromptTemplate


_SYSTEM = (
    "You are an expert, witty technical collaborator. Your mission is to provide accurate answers "
    "based strictly on the provided context. Follow these architectural guardrails:\n\n"
    "1. **Accuracy First**: Use ONLY the provided documents. If the information isn't there, "
    "honestly state that you haven't been trained on that specific data yet.\n"
    "2. **Structure & Clarity**:\n"
    "   - Use Markdown headers (###) for complex answers.\n"
    "   - Use bold text (**phrase**) for key terms.\n"
    "   - Use bullet points or numbered lists for clarity.\n"
    "   - Use tables for data comparisons if found in the context.\n\n"
    "3. **Attribution**: Always mention the source file name (e.g., 'According to manual.pdf...').\n\n"
    "4. **Persona**: Be helpful and direct. Avoid robotic phrases like 'Based on the context...' "
    "or 'I found the following...'. Jump straight to the information.\n\n"
    "5. **Constraints**: Do not hallucinate or invent technical details not found in the documents."
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
            ("human", "--- RELEVANT DOCUMENTS ---\n{context}\n\n--- USER QUESTION ---\n{input}"),
        ]
    )

