"""
Shared application state.
Holds global service instances to avoid circular imports between main.py and routes.
"""

from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.conversation import ConversationMemory
from app.services.rag import RAGService
from app.vectorstore.chroma_store import ChromaStore


class AppState:
    """Container for all application service instances."""
    embedding_service: EmbeddingService = None
    llm_service: LLMService = None
    vector_store: ChromaStore = None
    conversation_memory: ConversationMemory = None
    rag_service: RAGService = None


# Singleton state instance
state = AppState()
