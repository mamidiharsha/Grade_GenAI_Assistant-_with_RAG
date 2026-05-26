"""
Health check endpoint.
Returns system status and vector store statistics.
"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.state import state
from datetime import datetime

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns the current status of the application and vector store stats.
    """
    total_chunks = state.vector_store.count if state.vector_store else 0

    return HealthResponse(
        status="healthy",
        totalChunks=total_chunks,
        timestamp=datetime.utcnow().isoformat(),
    )
