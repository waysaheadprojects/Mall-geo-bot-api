"""Pydantic response models for the Statistical Analysis Bot API."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


class StatusEnum(str, Enum):
    """Status enumeration for API responses."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class BaseResponse(BaseModel):
    """Base response model with common fields."""
    status: StatusEnum = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")


class ErrorResponse(BaseResponse):
    """Error response model."""
    status: StatusEnum = Field(default=StatusEnum.ERROR)
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)
    uptime: Optional[str] = Field(None, description="Service uptime")


class DataUploadResponse(BaseResponse):
    """Data upload response."""
    data_info: Dict[str, Any] = Field(..., description="Information about uploaded data")
    quality_report: Dict[str, Any] = Field(..., description="Data quality assessment")
    processing_time: float = Field(..., description="Processing time in seconds")


class DataInfoResponse(BaseResponse):
    """Data information response."""
    basic_info: Dict[str, Any] = Field(..., description="Basic dataset information")
    schema_overview: Dict[str, Any] = Field(..., description="Schema and column information")
    storage_info: Dict[str, Any] = Field(..., description="Storage information")


class ColumnInfo(BaseModel):
    """Column information model."""
    column: str = Field(..., description="Column name")
    dtype: str = Field(..., description="Data type")
    null_count: int = Field(..., description="Number of null values")
    null_percentage: float = Field(..., description="Percentage of null values")
    unique_count: int = Field(..., description="Number of unique values")
    unique_percentage: float = Field(..., description="Percentage of unique values")
    is_numeric: bool = Field(..., description="Whether column is numeric")


class ColumnInfoResponse(BaseResponse):
    """Column information response."""
    column_info: ColumnInfo = Field(..., description="Column information")
    unique_values: List[Any] = Field(..., description="Sample of unique values")
    sample_values: List[Any] = Field(..., description="Sample values from the column")


class StatisticalSummary(BaseModel):
    """Statistical summary for a numeric column."""
    count: int = Field(..., description="Number of non-null values")
    mean: float = Field(..., description="Mean value")
    median: float = Field(..., description="Median value")
    mode: Optional[float] = Field(None, description="Mode value")
    std_dev: float = Field(..., description="Standard deviation")
    variance: float = Field(..., description="Variance")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    range: float = Field(..., description="Range (max - min)")
    q1: float = Field(..., description="First quartile")
    q3: float = Field(..., description="Third quartile")
    iqr: float = Field(..., description="Interquartile range")
    skewness: float = Field(..., description="Skewness")
    kurtosis: float = Field(..., description="Kurtosis")
    coef_var: Optional[float] = Field(None, description="Coefficient of variation")


class DescriptiveStatsResponse(BaseResponse):
    """Descriptive statistics response."""
    statistics: Dict[str, StatisticalSummary] = Field(..., description="Statistical summaries by column")
    analysis_summary: str = Field(..., description="Human-readable analysis summary")


class CorrelationPair(BaseModel):
    """Correlation pair information."""
    column1: str = Field(..., description="First column")
    column2: str = Field(..., description="Second column")
    correlation: float = Field(..., description="Correlation coefficient")
    strength: str = Field(..., description="Correlation strength interpretation")
    direction: str = Field(..., description="Correlation direction (positive/negative)")


class CorrelationAnalysisResponse(BaseResponse):
    """Correlation analysis response."""
    method: str = Field(..., description="Correlation method used")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(..., description="Full correlation matrix")
    strongest_correlations: List[CorrelationPair] = Field(..., description="Strongest correlation pairs")
    interpretation: str = Field(..., description="Analysis interpretation")
    summary: Dict[str, Any] = Field(..., description="Analysis summary statistics")


class OutlierInfo(BaseModel):
    """Outlier detection information."""
    count: int = Field(..., description="Number of outliers detected")
    percentage: float = Field(..., description="Percentage of outliers")
    outlier_values: List[float] = Field(..., description="Sample outlier values")


class IQROutlierInfo(OutlierInfo):
    """IQR-based outlier information."""
    lower_bound: float = Field(..., description="Lower bound for outlier detection")
    upper_bound: float = Field(..., description="Upper bound for outlier detection")


class ZScoreOutlierInfo(OutlierInfo):
    """Z-score-based outlier information."""
    threshold: float = Field(..., description="Z-score threshold used")


class OutlierDetectionResponse(BaseResponse):
    """Outlier detection response."""
    outliers: Dict[str, Dict[str, Union[IQROutlierInfo, ZScoreOutlierInfo]]] = Field(..., description="Outlier detection results by column and method")
    summary: str = Field(..., description="Analysis summary")


class NormalityTest(BaseModel):
    """Normality test results."""
    test: str = Field(..., description="Test name")
    statistic: float = Field(..., description="Test statistic")
    p_value: float = Field(..., description="P-value")
    is_normal: bool = Field(..., description="Whether data appears normally distributed")
    sample_size_used: int = Field(..., description="Sample size used for test")


class ShapeMetrics(BaseModel):
    """Distribution shape metrics."""
    skewness: float = Field(..., description="Skewness value")
    kurtosis: float = Field(..., description="Kurtosis value")
    interpretation: Dict[str, str] = Field(..., description="Interpretation of shape metrics")


class DistributionAnalysisResponse(BaseResponse):
    """Distribution analysis response."""
    column: str = Field(..., description="Analyzed column")
    sample_size: int = Field(..., description="Sample size")
    normality_test: NormalityTest = Field(..., description="Normality test results")
    shape_metrics: ShapeMetrics = Field(..., description="Distribution shape metrics")
    distribution_type: str = Field(..., description="Suggested distribution type")
    interpretation: str = Field(..., description="Overall interpretation")


class HypothesisTestResult(BaseModel):
    """Hypothesis test result."""
    test_type: str = Field(..., description="Type of hypothesis test performed")
    variables: List[str] = Field(..., description="Variables tested")
    statistic: float = Field(..., description="Test statistic")
    p_value: float = Field(..., description="P-value")
    is_significant: bool = Field(..., description="Whether result is statistically significant")
    interpretation: str = Field(..., description="Test interpretation")


class CorrelationTestResult(HypothesisTestResult):
    """Correlation test result."""
    correlation: float = Field(..., description="Correlation coefficient")
    sample_size: int = Field(..., description="Sample size used")


class ANOVATestResult(HypothesisTestResult):
    """ANOVA test result."""
    f_statistic: float = Field(..., description="F-statistic")
    groups: List[str] = Field(..., description="Group names")
    group_sizes: List[int] = Field(..., description="Sample sizes for each group")


class ChiSquareTestResult(HypothesisTestResult):
    """Chi-square test result."""
    chi2_statistic: float = Field(..., description="Chi-square statistic")
    degrees_of_freedom: int = Field(..., description="Degrees of freedom")
    contingency_table: Dict[str, Dict[str, int]] = Field(..., description="Contingency table")


class HypothesisTestResponse(BaseResponse):
    """Hypothesis test response."""
    test_result: Union[CorrelationTestResult, ANOVATestResult, ChiSquareTestResult] = Field(..., description="Test results")
    recommendations: List[str] = Field(..., description="Analysis recommendations")


class CategoricalSummary(BaseModel):
    """Categorical variable summary."""
    unique_values: int = Field(..., description="Number of unique values")
    most_frequent: Dict[str, Any] = Field(..., description="Most frequent value information")
    distribution: Dict[str, int] = Field(..., description="Value distribution (top values)")
    entropy: float = Field(..., description="Entropy measure")
    missing_values: int = Field(..., description="Number of missing values")


class CategoricalAnalysisResponse(BaseResponse):
    """Categorical analysis response."""
    analysis: Dict[str, CategoricalSummary] = Field(..., description="Categorical analysis by column")
    summary: str = Field(..., description="Analysis summary")


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class ChatQueryResponse(BaseResponse):
    """Chat query response."""
    response: str = Field(..., description="AI response to the query")
    conversation_history: Optional[List[ChatMessage]] = Field(None, description="Conversation history")
    suggested_questions: List[str] = Field(..., description="Suggested follow-up questions")
    analysis_performed: List[str] = Field(..., description="Statistical analyses performed")


class ChatHistoryResponse(BaseResponse):
    """Chat history response."""
    history: List[ChatMessage] = Field(..., description="Complete conversation history")
    message_count: int = Field(..., description="Total number of messages")


class SuggestionsResponse(BaseResponse):
    """Query suggestions response."""
    suggestions: List[str] = Field(..., description="Suggested questions based on the dataset")
    categories: Dict[str, List[str]] = Field(..., description="Suggestions grouped by category")

