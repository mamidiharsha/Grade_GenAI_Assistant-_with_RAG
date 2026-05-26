"""
Chat endpoint.
Handles user messages through the RAG pipeline.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.state import state
from app.utils.logger import logger

router = APIRouter(prefix="/api")


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Chat"],
)
async def chat(request: ChatRequest):
    """
    Process a chat message through the RAG pipeline.
    
    1. Validates the request payload
    2. Generates query embedding
    3. Retrieves relevant document chunks
    4. Builds prompt with context and history
    5. Generates LLM response
    6. Returns structured response
    """
    try:
        if not state.rag_service:
            raise HTTPException(
                status_code=503,
                detail="RAG service not initialized. Server is starting up.",
            )

        logger.info(
            f"Chat request | Session: {request.sessionId} | "
            f"Message: {request.message[:80]}..."
        )

        # Process through RAG pipeline
        response = state.rag_service.process_query(
            session_id=request.sessionId,
            message=request.message,
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Please try again.",
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later.",
        )
