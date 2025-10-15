"""
Custom exceptions for storage operations.
"""


class DuplicateImageError(Exception):
    """Raised when attempting to store a duplicate image."""

    def __init__(self, image_path: str, existing_id: str = None):
        self.image_path = image_path
        self.existing_id = existing_id
        message = f"Image already exists in database: {image_path}"
        if existing_id:
            message += f" (ID: {existing_id})"
        super().__init__(message)


class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass


class DatabaseConnectionError(StorageError):
    """Raised when database connection fails."""
    pass


class QueryError(StorageError):
    """Raised when a database query fails."""
    pass
