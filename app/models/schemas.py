"""
Pydantic models for request/response validation.
Ensures type safety and structured error handling across the API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


# ─── Request Models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Chat endpoint request payload."""
    sessionId: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique session identifier for conversation tracking",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="User's question or message",
    )

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        return v.strip()

    @field_validator("sessionId")
    @classmethod
    def session_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()


# ─── Response Models ──────────────────────────────────────────────

class ChatResponse(BaseModel):
    """Chat endpoint response payload."""
    reply: str = Field(..., description="Assistant's response")
    tokensUsed: int = Field(default=0, description="Total tokens consumed")
    retrievedChunks: int = Field(
        default=0,
        description="Number of relevant chunks retrieved",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    totalDocuments: Optional[int] = None
    totalChunks: Optional[int] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class ErrorResponse(BaseModel):
    """Structured error response."""
    error: str = Field(..., description="Error description")
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details",
    )


# ─── Internal Models ─────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A chunk retrieved from vector search."""
    text: str
    title: str
    chunk_id: str
    source: str
    similarity_score: float


class MessagePair(BaseModel):
    """A user-assistant message pair for conversation history."""
    user: str
    assistant: str
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
