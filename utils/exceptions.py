"""Custom exception classes for the Statistical Analysis Bot API."""

from typing import Any, Dict, Optional


class StatisticalBotException(Exception):
    """Base exception class for Statistical Bot errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class DataProcessingError(StatisticalBotException):
    """Raised when data processing fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class FileUploadError(StatisticalBotException):
    """Raised when file upload fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class AnalysisError(StatisticalBotException):
    """Raised when statistical analysis fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class LLMError(StatisticalBotException):
    """Raised when LLM processing fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=503, details=details)


class DataNotFoundError(StatisticalBotException):
    """Raised when requested data is not found."""
    
    def __init__(self, message: str = "No data found. Please upload data first.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=404, details=details)


class ValidationError(StatisticalBotException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)

