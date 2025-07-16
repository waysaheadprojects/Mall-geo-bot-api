"""Logging configuration for the Statistical Analysis Bot API."""

import logging
import sys
from typing import Dict, Any
from app.config import settings


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging():
    """Configure application logging."""
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))
    
    if settings.debug:
        formatter = ColoredFormatter(settings.log_format)
    else:
        formatter = logging.Formatter(settings.log_format)
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__name__)
    
    def log_operation(self, operation: str, **kwargs: Any) -> Dict[str, Any]:
        """Log an operation with context."""
        context = {
            "operation": operation,
            "class": self.__class__.__name__,
            **kwargs
        }
        self.logger.info(f"Operation: {operation}", extra=context)
        return context

