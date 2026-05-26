"""
Prompt templates for the RAG pipeline.
Defines the system prompt and context injection strategy.
"""


# System prompt that constrains the LLM to use only provided context
SYSTEM_PROMPT = """You are a helpful and professional AI assistant. Your role is to answer user questions accurately based ONLY on the provided context from the knowledge base.

STRICT RULES:
1. Answer ONLY based on the provided context below.
2. If the context does not contain enough information to answer the question, clearly state that you cannot find the information in the knowledge base.
3. Do NOT make up information or use knowledge outside the provided context.
4. Be concise, clear, and professional in your responses.
5. If the question is a follow-up, use the conversation history for continuity but still ground your answer in the context.
6. When referencing specific steps or procedures, present them in a clear, structured format."""


# Fallback response when similarity is below threshold
FALLBACK_RESPONSE = (
    "I could not find enough information in the knowledge base to answer "
    "this question. Please try rephrasing your question or ask about topics "
    "covered in our documentation."
)


def build_prompt(
    context: str,
    history: str,
    question: str,
) -> str:
    """
    Build the final prompt with retrieved context, conversation history,
    and the user's question.
    
    Args:
        context: Retrieved and formatted document chunks.
        history: Formatted conversation history string.
        question: The user's current question.
        
    Returns:
        Complete prompt string ready for LLM invocation.
    """
    prompt_parts = [SYSTEM_PROMPT]

    # Add retrieved context
    prompt_parts.append(f"\n--- Retrieved Context ---\n{context}")

    # Add conversation history if available
    if history and history.strip():
        prompt_parts.append(f"\n--- Conversation History ---\n{history}")

    # Add the current question
    prompt_parts.append(f"\n--- User Question ---\n{question}")

    # Add response instruction
    prompt_parts.append(
        "\n--- Instructions ---\n"
        "Provide a helpful answer based strictly on the context above. "
        "If the context doesn't contain relevant information, say so clearly."
    )

    return "\n".join(prompt_parts)


def format_context(chunks: list) -> str:
    """
    Format retrieved chunks into a structured context string.
    
    Args:
        chunks: List of RetrievedChunk objects.
        
    Returns:
        Formatted context string with numbered sources.
    """
    if not chunks:
        return "No relevant context found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = getattr(chunk, "source", "Unknown")
        text = getattr(chunk, "text", str(chunk))
        score = getattr(chunk, "similarity_score", 0.0)
        context_parts.append(
            f"[Source {i}: {source} | Relevance: {score:.2f}]\n{text}"
        )

    return "\n\n".join(context_parts)


def format_history(history: list) -> str:
    """
    Format conversation history into a readable string.
    
    Args:
        history: List of MessagePair objects.
        
    Returns:
        Formatted history string.
    """
    if not history:
        return ""

    lines = []
    for pair in history:
        user_msg = getattr(pair, "user", pair.get("user", ""))
        asst_msg = getattr(pair, "assistant", pair.get("assistant", ""))
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {asst_msg}")

    return "\n".join(lines)
