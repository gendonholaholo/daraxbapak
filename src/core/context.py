from typing import Dict, List, Optional
from .config import settings
from .logging import logger
from datetime import datetime

class ContextManager:
    def __init__(self):
        self.contexts: Dict[str, List[Dict]] = {}
        self.max_length = settings.MAX_CONTEXT_LENGTH
        self.compression_threshold = settings.CONTEXT_COMPRESSION_THRESHOLD
        self.metadata: Dict[str, Dict] = {}

    def add_context(self, session_id: str, context: Dict) -> None:
        """Add new context to the session"""
        if session_id not in self.contexts:
            self.contexts[session_id] = []
            self.metadata[session_id] = {
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "context_count": 0
            }
        
        # Add timestamp to context
        context["timestamp"] = datetime.now().isoformat()
        self.contexts[session_id].append(context)
        self.metadata[session_id]["context_count"] += 1
        self.metadata[session_id]["last_accessed"] = datetime.now()
        
        self._check_and_compress(session_id)

    def get_context(self, session_id: str) -> List[Dict]:
        """Get all contexts for a session"""
        if session_id in self.metadata:
            self.metadata[session_id]["last_accessed"] = datetime.now()
        return self.contexts.get(session_id, [])

    def clear_context(self, session_id: str) -> None:
        """Clear all contexts for a session"""
        if session_id in self.contexts:
            del self.contexts[session_id]
        if session_id in self.metadata:
            del self.metadata[session_id]

    def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """Get metadata for a session"""
        return self.metadata.get(session_id)

    def get_all_sessions(self) -> List[str]:
        """Get list of all active sessions"""
        return list(self.contexts.keys())

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        """Clean up sessions older than specified hours"""
        now = datetime.now()
        sessions_to_remove = []
        
        for session_id, metadata in self.metadata.items():
            age = now - metadata["last_accessed"]
            if age.total_seconds() > max_age_hours * 3600:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self.clear_context(session_id)
            logger.info(f"Cleaned up old session: {session_id}")

    def _check_and_compress(self, session_id: str) -> None:
        """Check context length and compress if necessary"""
        if session_id not in self.contexts:
            return

        current_length = sum(len(str(ctx)) for ctx in self.contexts[session_id])
        
        if current_length > self.compression_threshold:
            logger.info(f"Compressing context for session {session_id}")
            # Keep only the most recent contexts
            self.contexts[session_id] = self.contexts[session_id][-self.max_length:]
            self.metadata[session_id]["context_count"] = len(self.contexts[session_id])

context_manager = ContextManager() 