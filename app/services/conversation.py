"""
Conversation memory service.
Maintains short-term conversation history per session for context continuity.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from app.models.schemas import MessagePair
from app.config import get_settings
from app.utils.logger import logger


class ConversationMemory:
    """
    In-memory conversation store that maintains recent message history
    per session. Auto-cleans stale sessions to prevent memory leaks.
    """

    def __init__(self):
        self._store: Dict[str, List[MessagePair]] = {}
        self._last_access: Dict[str, datetime] = {}
        self._max_history = get_settings().MAX_HISTORY
        logger.info(
            f"Conversation memory initialized "
            f"(max_history={self._max_history})"
        )

    def add_message(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Add a user-assistant message pair to session history.
        
        Args:
            session_id: Unique session identifier.
            user_message: The user's message.
            assistant_message: The assistant's response.
        """
        if session_id not in self._store:
            self._store[session_id] = []

        pair = MessagePair(
            user=user_message,
            assistant=assistant_message,
        )
        self._store[session_id].append(pair)

        # Keep only the last N message pairs
        if len(self._store[session_id]) > self._max_history:
            self._store[session_id] = self._store[session_id][
                -self._max_history:
            ]

        self._last_access[session_id] = datetime.utcnow()

        logger.debug(
            f"Session '{session_id}': stored message pair "
            f"(total: {len(self._store[session_id])})"
        )

    def get_history(self, session_id: str) -> List[MessagePair]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Unique session identifier.
            
        Returns:
            List of MessagePair objects (empty if no history).
        """
        self._last_access[session_id] = datetime.utcnow()
        history = self._store.get(session_id, [])

        logger.debug(
            f"Session '{session_id}': retrieved {len(history)} message pairs"
        )
        return history

    def get_formatted_history(self, session_id: str) -> str:
        """
        Get conversation history formatted as a readable string.
        
        Args:
            session_id: Unique session identifier.
            
        Returns:
            Formatted history string for prompt injection.
        """
        history = self.get_history(session_id)

        if not history:
            return ""

        lines = []
        for pair in history:
            lines.append(f"User: {pair.user}")
            lines.append(f"Assistant: {pair.assistant}")

        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """
        Clear conversation history for a specific session.
        
        Args:
            session_id: Session to clear.
        """
        self._store.pop(session_id, None)
        self._last_access.pop(session_id, None)
        logger.info(f"Session '{session_id}': history cleared")

    def cleanup_stale_sessions(self, max_age_hours: int = 1) -> int:
        """
        Remove sessions that haven't been accessed recently.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup.
            
        Returns:
            Number of sessions cleaned up.
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        stale_sessions = [
            sid
            for sid, last in self._last_access.items()
            if last < cutoff
        ]

        for sid in stale_sessions:
            self.clear_session(sid)

        if stale_sessions:
            logger.info(
                f"Cleaned up {len(stale_sessions)} stale sessions"
            )

        return len(stale_sessions)

    @property
    def active_sessions(self) -> int:
        """Return the number of active sessions."""
        return len(self._store)
