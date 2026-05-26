"""
RAG (Retrieval-Augmented Generation) orchestrator.
Ties together embedding, vector search, prompt building, and LLM generation.
"""

import time
from typing import Tuple
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.conversation import ConversationMemory
from app.vectorstore.chroma_store import ChromaStore
from app.prompts.templates import (
    build_prompt,
    format_context,
    FALLBACK_RESPONSE,
)
from app.models.schemas import ChatResponse, RetrievedChunk
from app.config import get_settings
from app.utils.logger import logger


class RAGService:
    """
    Core RAG orchestrator that processes user queries through the
    full retrieval-augmented generation pipeline.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        vector_store: ChromaStore,
        conversation_memory: ConversationMemory,
    ):
        self._embedding = embedding_service
        self._llm = llm_service
        self._vector_store = vector_store
        self._memory = conversation_memory
        self._settings = get_settings()

        logger.info("RAG service initialized")

    def process_query(
        self,
        session_id: str,
        message: str,
    ) -> ChatResponse:
        """
        Process a user query through the complete RAG pipeline:
        1. Generate query embedding
        2. Perform similarity search
        3. Check similarity threshold
        4. Build prompt with context + history
        5. Generate LLM response
        6. Store in conversation memory
        
        Args:
            session_id: User's session identifier.
            message: The user's question.
            
        Returns:
            ChatResponse with reply, token count, and retrieved chunk count.
        """
        start_time = time.time()
        logger.info(f"Processing query for session '{session_id}': {message[:100]}...")

        # ─── Step 1: Generate query embedding ─────────────────────
        logger.info("Step 1: Generating query embedding...")
        query_embedding = self._embedding.embed_query(message)

        # ─── Step 2: Perform similarity search ────────────────────
        logger.info("Step 2: Performing similarity search...")
        retrieved_chunks = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=self._settings.TOP_K,
            similarity_threshold=self._settings.SIMILARITY_THRESHOLD,
        )

        # ─── Step 3: Check if sufficient context was found ────────
        if not retrieved_chunks:
            logger.warning(
                f"No chunks above threshold ({self._settings.SIMILARITY_THRESHOLD}) "
                f"for query: {message[:80]}..."
            )
            # Store fallback in conversation memory
            self._memory.add_message(
                session_id, message, FALLBACK_RESPONSE
            )
            return ChatResponse(
                reply=FALLBACK_RESPONSE,
                tokensUsed=0,
                retrievedChunks=0,
            )

        # ─── Step 4: Build prompt ─────────────────────────────────
        logger.info(
            f"Step 3: Building prompt with {len(retrieved_chunks)} chunks..."
        )

        # Format retrieved context
        context_str = format_context(retrieved_chunks)

        # Get conversation history
        history_str = self._memory.get_formatted_history(session_id)

        # Build the complete prompt
        prompt = build_prompt(
            context=context_str,
            history=history_str,
            question=message,
        )

        # ─── Step 5: Generate LLM response ───────────────────────
        logger.info("Step 4: Generating LLM response...")
        response_text, tokens_used = self._llm.generate_response(prompt)

        # ─── Step 6: Store in conversation memory ─────────────────
        self._memory.add_message(session_id, message, response_text)

        elapsed = time.time() - start_time
        logger.info(
            f"Query processed in {elapsed:.2f}s | "
            f"Chunks: {len(retrieved_chunks)} | "
            f"Tokens: {tokens_used}"
        )

        return ChatResponse(
            reply=response_text,
            tokensUsed=tokens_used,
            retrievedChunks=len(retrieved_chunks),
        )
