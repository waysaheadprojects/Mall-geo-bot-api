"""Data management endpoints for the Statistical Analysis Bot API."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import time

from app.dependencies import get_data_handler, get_current_data, get_optional_data, validate_column_exists
from app.api.models.requests import ColumnInfoRequest
from app.api.models.responses import (
    DataUploadResponse, DataInfoResponse, BaseResponse, StatusEnum, 
    ColumnInfoResponse, ColumnInfo
)
from core.storage import storage
from core.data_handler import DataHandler
from utils.exceptions import FileUploadError, DataProcessingError
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/data", tags=["Data Management"])


@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(
    files: List[UploadFile] = File(..., description="CSV or Excel files to upload"),
    data_handler: DataHandler = Depends(get_data_handler)
):
    """
    Upload and process data files.
    
    Supports CSV and Excel files. Multiple files will be combined into a single dataset.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing {len(files)} uploaded files")
        
        # Process uploaded files
        df = await data_handler.process_uploaded_files(files)
        
        # Get data quality report
        quality_report = data_handler.validate_data_quality(df)
        
        # Get column information
        column_info = data_handler.get_column_info(df)
        
        # Prepare metadata
        metadata = {
            "upload_time": time.time(),
            "file_names": [file.filename for file in files],
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "quality_report": quality_report,
            "column_info": column_info
        }
        
        # Save to storage
        if not storage.save_data(df, metadata):
            raise DataProcessingError("Failed to save processed data")
        
        processing_time = time.time() - start_time
        
        logger.info(f"Data upload completed in {processing_time:.2f} seconds")
        
        return DataUploadResponse(
            status=StatusEnum.SUCCESS,
            message=f"Successfully processed {len(files)} files with {len(df)} rows and {len(df.columns)} columns",
            data_info={
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": len(column_info['numeric_columns']),
                "categorical_columns": len(column_info['categorical_columns']),
                "file_names": [file.filename for file in files]
            },
            quality_report=quality_report,
            processing_time=processing_time
        )
        
    except FileUploadError as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    except DataProcessingError as e:
        logger.error(f"Data processing error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during upload")


@router.get("/info", response_model=DataInfoResponse)
async def get_data_info(data: tuple = Depends(get_current_data)):
    """
    Get information about the currently loaded dataset.
    """
    df, metadata = data
    
    try:
        # Get basic info
        basic_info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "upload_time": metadata.get("upload_time"),
            "file_names": metadata.get("file_names", [])
        }
        
        # Get schema overview
        data_handler = DataHandler()
        schema_overview = data_handler.get_column_info(df)
        
        # Get storage info
        storage_info = storage.get_storage_stats()
        
        return DataInfoResponse(
            status=StatusEnum.SUCCESS,
            message="Dataset information retrieved successfully",
            basic_info=basic_info,
            schema_overview=schema_overview,
            storage_info=storage_info
        )
        
    except Exception as e:
        logger.error(f"Error getting data info: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve data information")


@router.delete("/clear", response_model=BaseResponse)
async def clear_data():
    """
    Clear all stored data and chat history.
    """
    try:
        if storage.clear_data():
            logger.info("Data storage cleared successfully")
            return BaseResponse(
                status=StatusEnum.SUCCESS,
                message="All data and chat history cleared successfully"
            )
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear data")
            
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear data")


@router.get("/columns", response_model=Dict[str, Any])
async def get_columns(data: tuple = Depends(get_current_data)):
    """
    Get list of all columns with their types and basic information.
    """
    df, _ = data
    
    try:
        data_handler = DataHandler()
        column_info = data_handler.get_column_info(df)
        
        # Add sample values for each column
        columns_detail = []
        for col in df.columns:
            col_data = df[col].dropna()
            sample_values = col_data.head(5).tolist() if len(col_data) > 0 else []
            
            columns_detail.append({
                "name": col,
                "type": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float((df[col].isnull().sum() / len(df)) * 100),
                "unique_count": int(df[col].nunique()),
                "sample_values": sample_values,
                "is_numeric": col in column_info['numeric_columns']
            })
        
        return {
            "status": "success",
            "message": "Column information retrieved successfully",
            "columns": columns_detail,
            "summary": {
                "total_columns": len(df.columns),
                "numeric_columns": column_info['numeric_columns'],
                "categorical_columns": column_info['categorical_columns']
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting columns: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve column information")


@router.get("/columns/{column_name}", response_model=ColumnInfoResponse)
async def get_column_info(
    column_name: str,
    limit: int = 100,
    data: tuple = Depends(get_current_data)
):
    """
    Get detailed information about a specific column.
    """
    df, _ = data
    
    # Validate column exists
    if column_name not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Column '{column_name}' not found"
        )
    
    try:
        data_handler = DataHandler()
        
        # Get basic column info
        col_data = df[column_name]
        dtype = str(col_data.dtype)
        null_count = int(col_data.isnull().sum())
        null_percentage = float((null_count / len(df)) * 100)
        unique_count = int(col_data.nunique())
        unique_percentage = float((unique_count / len(df)) * 100)
        
        # Determine if numeric
        column_info_dict = data_handler.get_column_info(df)
        is_numeric = column_name in column_info_dict['numeric_columns']
        
        column_info = ColumnInfo(
            column=column_name,
            dtype=dtype,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            is_numeric=is_numeric
        )
        
        # Get unique values (limited)
        unique_values = data_handler.get_column_unique_values(df, column_name, limit)
        
        # Get sample values
        sample_values = col_data.dropna().head(10).tolist()
        
        return ColumnInfoResponse(
            status=StatusEnum.SUCCESS,
            message=f"Column information for '{column_name}' retrieved successfully",
            column_info=column_info,
            unique_values=unique_values,
            sample_values=sample_values
        )
        
    except Exception as e:
        logger.error(f"Error getting column info for {column_name}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve information for column '{column_name}'")


@router.get("/status")
async def get_data_status():
    """
    Get current data status (whether data is loaded, storage info, etc.).
    """
    try:
        data_exists = storage.data_exists()
        storage_stats = storage.get_storage_stats()
        
        status_info = {
            "data_loaded": data_exists,
            "storage_stats": storage_stats
        }
        
        if data_exists:
            metadata = storage.get_data_info()
            if metadata:
                status_info["data_summary"] = {
                    "rows": metadata.get("data_shape", [0, 0])[0],
                    "columns": metadata.get("data_shape", [0, 0])[1],
                    "upload_time": metadata.get("upload_time"),
                    "file_names": metadata.get("file_names", [])
                }
        
        return {
            "status": "success",
            "message": "Data status retrieved successfully",
            "data": status_info
        }
        
    except Exception as e:
        logger.error(f"Error getting data status: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve data status")

