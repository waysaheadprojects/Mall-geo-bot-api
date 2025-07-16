"""Data storage and persistence module for the Statistical Analysis Bot API."""

import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

from app.config import settings
from utils.exceptions import DataProcessingError
from utils.logging import LoggerMixin


class DataStorage(LoggerMixin):
    """Handles data persistence and retrieval."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or settings.data_storage_dir
        self.storage_dir.mkdir(exist_ok=True)
        
        self.data_file = self.storage_dir / "processed_data.pkl"
        self.metadata_file = self.storage_dir / "metadata.json"
        self.chat_history_file = self.storage_dir / "chat_history.json"
    
    def save_data(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
        """Save processed data and metadata to persistent storage."""
        self.log_operation("save_data", rows=len(df), columns=len(df.columns))
        
        try:
            # Add timestamp to metadata
            metadata["saved_at"] = datetime.now().isoformat()
            metadata["data_shape"] = [len(df), len(df.columns)]
            
            # Save DataFrame using pickle for efficiency
            with open(self.data_file, "wb") as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata as JSON for readability
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Data saved successfully: {len(df)} rows, {len(df.columns)} columns")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")
            raise DataProcessingError(f"Failed to save data: {str(e)}")
    
    def load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Load processed data and metadata from persistent storage."""
        self.log_operation("load_data")
        
        if not self.data_file.exists() or not self.metadata_file.exists():
            self.logger.info("No stored data found")
            return None, None
        
        try:
            # Load DataFrame
            with open(self.data_file, "rb") as f:
                df = pickle.load(f)
            
            # Load metadata
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
            
            self.logger.info(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            return df, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise DataProcessingError(f"Failed to load data: {str(e)}")
    
    def clear_data(self) -> bool:
        """Clear stored data files."""
        self.log_operation("clear_data")
        
        try:
            files_removed = 0
            
            if self.data_file.exists():
                self.data_file.unlink()
                files_removed += 1
            
            if self.metadata_file.exists():
                self.metadata_file.unlink()
                files_removed += 1
            
            if self.chat_history_file.exists():
                self.chat_history_file.unlink()
                files_removed += 1
            
            self.logger.info(f"Cleared {files_removed} storage files")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear data: {str(e)}")
            raise DataProcessingError(f"Failed to clear data: {str(e)}")
    
    def data_exists(self) -> bool:
        """Check if stored data exists."""
        return self.data_file.exists() and self.metadata_file.exists()
    
    def get_data_info(self) -> Optional[Dict[str, Any]]:
        """Get information about stored data without loading it."""
        if not self.metadata_file.exists():
            return None
        
        try:
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to read metadata: {str(e)}")
            return None
    
    def save_chat_history(self, chat_history: list) -> bool:
        """Save chat history to persistent storage."""
        self.log_operation("save_chat_history", messages=len(chat_history))
        
        try:
            chat_data = {
                "history": chat_history,
                "saved_at": datetime.now().isoformat(),
                "message_count": len(chat_history)
            }
            
            with open(self.chat_history_file, "w") as f:
                json.dump(chat_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save chat history: {str(e)}")
            return False
    
    def load_chat_history(self) -> list:
        """Load chat history from persistent storage."""
        if not self.chat_history_file.exists():
            return []
        
        try:
            with open(self.chat_history_file, "r") as f:
                chat_data = json.load(f)
            
            return chat_data.get("history", [])
            
        except Exception as e:
            self.logger.error(f"Failed to load chat history: {str(e)}")
            return []
    
    def clear_chat_history(self) -> bool:
        """Clear stored chat history."""
        self.log_operation("clear_chat_history")
        
        try:
            if self.chat_history_file.exists():
                self.chat_history_file.unlink()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear chat history: {str(e)}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "storage_directory": str(self.storage_dir),
            "data_exists": self.data_exists(),
            "chat_history_exists": self.chat_history_file.exists(),
            "files": {}
        }
        
        # Check file sizes
        for file_path, name in [
            (self.data_file, "data_file"),
            (self.metadata_file, "metadata_file"),
            (self.chat_history_file, "chat_history_file")
        ]:
            if file_path.exists():
                stats["files"][name] = {
                    "exists": True,
                    "size_bytes": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            else:
                stats["files"][name] = {"exists": False}
        
        return stats


# Global storage instance
storage = DataStorage()

