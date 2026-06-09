"""
Custom exceptions for PostureKit with rich context tracking.

All exceptions inherit from PostureKitError which captures:
- Operation being performed
- Parameters involved
- Timestamp of failure
- Original exception (if wrapped)
"""

from typing import Dict, Any, Optional
from datetime import datetime
import traceback


class PostureKitError(Exception):
    """Base exception for all PostureKit errors with context tracking."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.operation = operation or "unknown"
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.now().isoformat()
        self.traceback = traceback.format_exc() if original_error else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'operation': self.operation,
            'context': self.context,
            'timestamp': self.timestamp,
            'traceback': self.traceback,
            'original_error': str(self.original_error) if self.original_error else None
        }

    def __str__(self) -> str:
        parts = [f"{self.__class__.__name__}: {self.message}"]
        if self.operation != "unknown":
            parts.append(f"Operation: {self.operation}")
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.original_error:
            parts.append(f"Caused by: {self.original_error}")
        return " | ".join(parts)


class DetectionError(PostureKitError):
    """Raised when pose detection fails."""
    pass


class StorageError(PostureKitError):
    """Raised when database storage operations fail."""
    pass


class SearchError(PostureKitError):
    """Raised when similarity search operations fail."""
    pass


class ValidationError(PostureKitError):
    """Raised when input validation fails."""
    pass


class IndexError(PostureKitError):
    """Raised when FAISS index operations fail."""
    pass


class ConfigurationError(PostureKitError):
    """Raised when configuration is invalid."""
    pass
