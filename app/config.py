import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    app_name: str = "Statistical Analysis Bot API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    
    # Data Storage
    data_storage_dir: Path = Path("data_storage")
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: list = [".csv", ".xlsx", ".xls"]
    
    # CORS Configuration
    cors_origins: list = ["*"]
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Ensure data storage directory exists
settings.data_storage_dir.mkdir(exist_ok=True)

