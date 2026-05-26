"""
GenAI Assistant with RAG - Main Application
=============================================
FastAPI application entry point.
Handles startup initialization, CORS, static files, and route registration.
"""

import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.config import get_settings
from app.utils.logger import logger
from app.utils.chunker import chunk_all_documents
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.conversation import ConversationMemory
from app.services.rag import RAGService
from app.vectorstore.chroma_store import ChromaStore
from app.state import state
from app.routes import chat, health


def load_documents(filepath: str = "docs.json") -> list:
    """
    Load documents from the JSON knowledge base file.
    
    Args:
        filepath: Path to the docs.json file.
        
    Returns:
        List of document dictionaries.
    """
    if not os.path.exists(filepath):
        logger.error(f"Knowledge base file not found: {filepath}")
        raise FileNotFoundError(
            f"docs.json not found at {filepath}. "
            "Please create the knowledge base file."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        documents = json.load(f)

    logger.info(f"Loaded {len(documents)} documents from {filepath}")
    return documents


async def initialize_rag_pipeline():
    """
    Initialize the complete RAG pipeline on application startup:
    1. Load documents from docs.json
    2. Chunk documents into smaller pieces
    3. Generate embeddings for all chunks
    4. Store embeddings in ChromaDB
    5. Initialize all services
    """
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("Initializing RAG Pipeline...")
    logger.info("=" * 60)

    # Step 1: Initialize services
    logger.info("Step 1: Initializing services...")
    state.embedding_service = EmbeddingService()
    state.llm_service = LLMService()
    state.vector_store = ChromaStore(persist_dir="./chroma_db")
    state.conversation_memory = ConversationMemory()

    # Step 2: Load and chunk documents
    logger.info("Step 2: Loading documents...")
    documents = load_documents("docs.json")

    logger.info("Step 3: Chunking documents...")
    chunks = chunk_all_documents(
        documents,
        max_tokens=settings.CHUNK_SIZE,
        overlap_tokens=settings.CHUNK_OVERLAP,
    )

    # Step 3: Generate embeddings and store
    # Always re-index to ensure embedding consistency
    logger.info(
        f"Step 4: Generating embeddings for {len(chunks)} chunks..."
    )
    chunk_texts = [c["text"] for c in chunks]
    embeddings = state.embedding_service.embed_batch(chunk_texts)

    logger.info("Step 5: Storing in ChromaDB...")
    state.vector_store.reset()  # Clear old embeddings to avoid dimension mismatch
    state.vector_store.add_documents(chunks, embeddings)

    # Step 4: Initialize RAG orchestrator
    state.rag_service = RAGService(
        embedding_service=state.embedding_service,
        llm_service=state.llm_service,
        vector_store=state.vector_store,
        conversation_memory=state.conversation_memory,
    )

    logger.info("=" * 60)
    logger.info(
        f"RAG Pipeline Ready! "
        f"({state.vector_store.count} chunks indexed)"
    )
    logger.info("=" * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    await initialize_rag_pipeline()
    yield
    # Shutdown
    logger.info("Application shutting down...")
    if state.conversation_memory:
        state.conversation_memory.cleanup_stale_sessions(max_age_hours=0)


# ─── Create FastAPI Application ──────────────────────────────────
app = FastAPI(
    title="GenAI Assistant with RAG",
    description=(
        "A production-grade AI chat assistant powered by "
        "Retrieval-Augmented Generation (RAG) using Google Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS Middleware ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register Routes ─────────────────────────────────────────────
app.include_router(chat.router)
app.include_router(health.router)

# ─── Mount Frontend Static Files ─────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ─── Root Route (Serve Frontend) ─────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend HTML file."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        {"message": "GenAI Assistant API is running. Visit /docs for API documentation."}
    )


# ─── Global Exception Handler ────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
