"""Pydantic request models for the Statistical Analysis Bot API."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Any, Dict
from enum import Enum


class AnalysisType(str, Enum):
    """Available analysis types."""
    DESCRIPTIVE = "descriptive"
    CORRELATION = "correlation"
    OUTLIERS = "outliers"
    DISTRIBUTION = "distribution"
    HYPOTHESIS = "hypothesis"


class CorrelationMethod(str, Enum):
    """Correlation analysis methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class OutlierMethod(str, Enum):
    """Outlier detection methods."""
    IQR = "iqr"
    ZSCORE = "zscore"
    BOTH = "both"


class HypothesisTestType(str, Enum):
    """Hypothesis test types."""
    AUTO = "auto"
    CORRELATION = "correlation"
    ANOVA = "anova"
    CHI_SQUARE = "chi_square"


class ChatQueryRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query about the data")
    include_history: bool = Field(default=True, description="Whether to include conversation history in response")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class DescriptiveStatsRequest(BaseModel):
    """Request model for descriptive statistics."""
    column: Optional[str] = Field(None, description="Specific column to analyze (if None, analyzes all numeric columns)")
    
    @validator('column')
    def validate_column(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip() if v else None


class CorrelationAnalysisRequest(BaseModel):
    """Request model for correlation analysis."""
    method: CorrelationMethod = Field(default=CorrelationMethod.PEARSON, description="Correlation method to use")
    columns: Optional[List[str]] = Field(None, description="Specific columns to include (if None, uses all numeric columns)")
    
    @validator('columns')
    def validate_columns(cls, v):
        if v is not None:
            if len(v) < 2:
                raise ValueError("Need at least 2 columns for correlation analysis")
            return [col.strip() for col in v if col.strip()]
        return v


class OutlierDetectionRequest(BaseModel):
    """Request model for outlier detection."""
    column: Optional[str] = Field(None, description="Specific column to analyze (if None, analyzes all numeric columns)")
    methods: List[OutlierMethod] = Field(default=[OutlierMethod.IQR], description="Outlier detection methods to use")
    
    @validator('column')
    def validate_column(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip() if v else None
    
    @validator('methods')
    def validate_methods(cls, v):
        if not v:
            raise ValueError("At least one method must be specified")
        # Convert "both" to individual methods
        if OutlierMethod.BOTH in v:
            return [OutlierMethod.IQR, OutlierMethod.ZSCORE]
        return v


class DistributionAnalysisRequest(BaseModel):
    """Request model for distribution analysis."""
    column: str = Field(..., min_length=1, description="Column to analyze distribution for")
    
    @validator('column')
    def validate_column(cls, v):
        if not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip()


class HypothesisTestRequest(BaseModel):
    """Request model for hypothesis testing."""
    column1: str = Field(..., min_length=1, description="First variable for hypothesis test")
    column2: str = Field(..., min_length=1, description="Second variable for hypothesis test")
    test_type: HypothesisTestType = Field(default=HypothesisTestType.AUTO, description="Type of hypothesis test")
    
    @validator('column1', 'column2')
    def validate_columns(cls, v):
        if not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip()
    
    @validator('column2')
    def validate_different_columns(cls, v, values):
        if 'column1' in values and v == values['column1']:
            raise ValueError("Column1 and column2 must be different")
        return v


class ColumnInfoRequest(BaseModel):
    """Request model for column information."""
    column: str = Field(..., min_length=1, description="Column name to get information for")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of unique values to return")
    
    @validator('column')
    def validate_column(cls, v):
        if not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip()


class CategoricalAnalysisRequest(BaseModel):
    """Request model for categorical analysis."""
    column: Optional[str] = Field(None, description="Specific column to analyze (if None, analyzes all categorical columns)")
    
    @validator('column')
    def validate_column(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip() if v else None

