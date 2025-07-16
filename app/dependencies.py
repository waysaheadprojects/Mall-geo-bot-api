"""Dependency injection for the Statistical Analysis Bot API."""

from fastapi import Depends, HTTPException, status
from typing import Tuple, Optional
import pandas as pd

from core.storage import storage
from core.data_handler import DataHandler
from core.statistical_analyzer import StatisticalAnalyzer
from core.llm_agent import StatisticalLLMAgent
from utils.exceptions import DataNotFoundError
from utils.logging import get_logger

logger = get_logger(__name__)


def get_data_handler() -> DataHandler:
    """Get DataHandler instance."""
    return DataHandler()


def get_current_data() -> Tuple[pd.DataFrame, dict]:
    """Get currently loaded data and metadata."""
    df, metadata = storage.load_data()
    
    if df is None or metadata is None:
        logger.warning("No data found in storage")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No data found. Please upload data first."
        )
    
    logger.info(f"Retrieved data: {len(df)} rows, {len(df.columns)} columns")
    return df, metadata


def get_statistical_analyzer(data: Tuple[pd.DataFrame, dict] = Depends(get_current_data)) -> StatisticalAnalyzer:
    """Get StatisticalAnalyzer instance with current data."""
    df, _ = data
    return StatisticalAnalyzer(df)


def get_llm_agent(
    data: Tuple[pd.DataFrame, dict] = Depends(get_current_data),
    analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)
) -> StatisticalLLMAgent:
    """Get StatisticalLLMAgent instance with current data and analyzer."""
    df, _ = data
    return StatisticalLLMAgent(df, analyzer)


def check_data_exists() -> bool:
    """Check if data exists in storage."""
    return storage.data_exists()


def get_optional_data() -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Get data if it exists, otherwise return None."""
    try:
        return get_current_data()
    except HTTPException:
        return None, None


def validate_column_exists(column: str, data: Tuple[pd.DataFrame, dict] = Depends(get_current_data)) -> str:
    """Validate that a column exists in the current dataset."""
    df, _ = data
    
    if column not in df.columns:
        available_columns = list(df.columns)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{column}' not found. Available columns: {available_columns}"
        )
    
    return column


def validate_numeric_column(column: str, analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)) -> str:
    """Validate that a column is numeric."""
    if column not in analyzer.numeric_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{column}' is not numeric. Numeric columns: {analyzer.numeric_columns}"
        )
    
    return column


def validate_categorical_column(column: str, analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)) -> str:
    """Validate that a column is categorical."""
    if column not in analyzer.categorical_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{column}' is not categorical. Categorical columns: {analyzer.categorical_columns}"
        )
    
    return column


def validate_columns_for_correlation(
    columns: list, 
    analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)
) -> list:
    """Validate that columns are suitable for correlation analysis."""
    if len(columns) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 columns for correlation analysis"
        )
    
    # Check if all columns are numeric
    non_numeric = [col for col in columns if col not in analyzer.numeric_columns]
    if non_numeric:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Non-numeric columns found: {non_numeric}. Only numeric columns can be used for correlation."
        )
    
    return columns


def validate_columns_for_hypothesis_test(
    column1: str, 
    column2: str, 
    data: Tuple[pd.DataFrame, dict] = Depends(get_current_data)
) -> Tuple[str, str]:
    """Validate columns for hypothesis testing."""
    df, _ = data
    
    # Check if columns exist
    for col in [column1, column2]:
        if col not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{col}' not found in dataset"
            )
    
    # Check if columns are different
    if column1 == column2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Column1 and column2 must be different"
        )
    
    return column1, column2

