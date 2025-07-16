"""Statistical analysis endpoints for the Statistical Analysis Bot API."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List

from app.dependencies import (
    get_statistical_analyzer, validate_column_exists, validate_numeric_column,
    validate_categorical_column, validate_columns_for_correlation, validate_columns_for_hypothesis_test
)
from app.api.models.requests import (
    DescriptiveStatsRequest, CorrelationAnalysisRequest, OutlierDetectionRequest,
    DistributionAnalysisRequest, HypothesisTestRequest, CategoricalAnalysisRequest
)
from app.api.models.responses import (
    DescriptiveStatsResponse, CorrelationAnalysisResponse, OutlierDetectionResponse,
    DistributionAnalysisResponse, HypothesisTestResponse, CategoricalAnalysisResponse,
    StatusEnum, StatisticalSummary, CorrelationPair
)
from core.statistical_analyzer import StatisticalAnalyzer
from utils.exceptions import AnalysisError, ValidationError
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["Statistical Analysis"])


@router.post("/descriptive", response_model=DescriptiveStatsResponse)
async def get_descriptive_statistics(
    request: DescriptiveStatsRequest,
    analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)
):
    """
    Calculate descriptive statistics for numeric columns.
    
    If no column is specified, analyzes all numeric columns.
    """
    try:
        logger.info(f"Computing descriptive statistics for column: {request.column}")
        
        # Validate column if specified
        if request.column:
            if request.column not in analyzer.numeric_columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{request.column}' is not numeric. Available numeric columns: {analyzer.numeric_columns}"
                )
        
        # Calculate statistics
        stats_results = analyzer.descriptive_statistics(request.column)
        
        # Convert to response format
        statistics = {}
        for col, stats in stats_results.items():
            statistics[col] = StatisticalSummary(**stats)
        
        # Generate analysis summary
        if request.column:
            col_stats = stats_results[request.column]
            summary = f"Descriptive statistics for '{request.column}': Mean = {col_stats['mean']:.2f}, "
            summary += f"Std Dev = {col_stats['std_dev']:.2f}, Range = [{col_stats['min']:.2f}, {col_stats['max']:.2f}]"
        else:
            summary = f"Descriptive statistics calculated for {len(statistics)} numeric columns. "
            summary += f"Columns analyzed: {', '.join(statistics.keys())}"
        
        return DescriptiveStatsResponse(
            status=StatusEnum.SUCCESS,
            message="Descriptive statistics calculated successfully",
            statistics=statistics,
            analysis_summary=summary
        )
        
    except AnalysisError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error calculating descriptive statistics: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to calculate descriptive statistics")


@router.post("/correlation", response_model=CorrelationAnalysisResponse)
async def get_correlation_analysis(
    request: CorrelationAnalysisRequest,
    analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)
):
    """
    Perform correlation analysis between numeric variables.
    
    If no columns are specified, analyzes all numeric columns.
    """
    try:
        logger.info(f"Computing correlation analysis with method: {request.method}")
        
        # Validate columns if specified
        if request.columns:
            if len(request.columns) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Need at least 2 columns for correlation analysis"
                )
            
            # Check if all columns are numeric
            non_numeric = [col for col in request.columns if col not in analyzer.numeric_columns]
            if non_numeric:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Non-numeric columns found: {non_numeric}. Available numeric columns: {analyzer.numeric_columns}"
                )
        
        # Perform correlation analysis
        correlation_results = analyzer.correlation_analysis(method=request.method.value)
        
        # Convert correlation pairs to response format
        strongest_correlations = []
        for pair in correlation_results['strongest_correlations']:
            strongest_correlations.append(CorrelationPair(**pair))
        
        return CorrelationAnalysisResponse(
            status=StatusEnum.SUCCESS,
            message=f"Correlation analysis completed using {request.method.value} method",
            method=correlation_results['method'],
            correlation_matrix=correlation_results['correlation_matrix'],
            strongest_correlations=strongest_correlations,
            interpretation=correlation_results['interpretation'],
            summary=correlation_results['summary']
        )
        
    except AnalysisError as e:
        logger.error(f"Correlation analysis error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error performing correlation analysis: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to perform correlation analysis")


@router.post("/outliers", response_model=OutlierDetectionResponse)
async def detect_outliers(
    request: OutlierDetectionRequest,
    analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)
):
    """
    Detect outliers in numeric columns using specified methods.
    
    If no column is specified, analyzes all numeric columns.
    """
    try:
        logger.info(f"Detecting outliers for column: {request.column}, methods: {request.methods}")
        
        # Validate column if specified
        if request.column:
            if request.column not in analyzer.numeric_columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{request.column}' is not numeric. Available numeric columns: {analyzer.numeric_columns}"
                )
        
        # Convert methods to strings
        methods = [method.value for method in request.methods]
        
        # Detect outliers
        outlier_results = analyzer.outlier_detection(request.column, methods)
        
        # Generate summary
        if request.column:
            col_results = outlier_results[request.column]
            summary = f"Outlier detection for '{request.column}': "
            for method, results in col_results.items():
                summary += f"{method.upper()}: {results['count']} outliers ({results['percentage']:.1f}%); "
        else:
            total_outliers = sum(
                sum(method_results['count'] for method_results in col_results.values())
                for col_results in outlier_results.values()
            )
            summary = f"Outlier detection completed for {len(outlier_results)} columns. Total outliers found: {total_outliers}"
        
        return OutlierDetectionResponse(
            status=StatusEnum.SUCCESS,
            message="Outlier detection completed successfully",
            outliers=outlier_results,
            summary=summary
        )
        
    except AnalysisError as e:
        logger.error(f"Outlier detection error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to detect outliers")


@router.post("/distribution", response_model=DistributionAnalysisResponse)
async def analyze_distribution(
    request: DistributionAnalysisRequest,
    analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)
):
    """
    Analyze the distribution characteristics of a numeric column.
    """
    try:
        logger.info(f"Analyzing distribution for column: {request.column}")
        
        # Validate column
        if request.column not in analyzer.numeric_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{request.column}' is not numeric. Available numeric columns: {analyzer.numeric_columns}"
            )
        
        # Analyze distribution
        distribution_results = analyzer.distribution_analysis(request.column)
        
        # Generate interpretation
        normality = distribution_results['normality_test']
        shape = distribution_results['shape_metrics']
        
        interpretation = f"Distribution analysis for '{request.column}': "
        interpretation += f"{'Normal' if normality['is_normal'] else 'Non-normal'} distribution "
        interpretation += f"(Shapiro-Wilk p-value: {normality['p_value']:.4f}). "
        interpretation += f"Shape: {shape['interpretation']['skewness_interpretation']}, "
        interpretation += f"{shape['interpretation']['kurtosis_interpretation']}. "
        interpretation += f"Suggested type: {distribution_results['distribution_type']}"
        
        return DistributionAnalysisResponse(
            status=StatusEnum.SUCCESS,
            message=f"Distribution analysis completed for '{request.column}'",
            column=distribution_results['column'],
            sample_size=distribution_results['sample_size'],
            normality_test=distribution_results['normality_test'],
            shape_metrics=distribution_results['shape_metrics'],
            distribution_type=distribution_results['distribution_type'],
            interpretation=interpretation
        )
        
    except AnalysisError as e:
        logger.error(f"Distribution analysis error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error analyzing distribution: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to analyze distribution")


@router.post("/hypothesis", response_model=HypothesisTestResponse)
async def perform_hypothesis_test(
    request: HypothesisTestRequest,
    analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)
):
    """
    Perform hypothesis testing between two variables.
    
    The test type is automatically determined based on variable types unless specified.
    """
    try:
        logger.info(f"Performing hypothesis test: {request.column1} vs {request.column2}")
        
        # Validate columns exist
        for col in [request.column1, request.column2]:
            if col not in analyzer.df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{col}' not found in dataset"
                )
        
        # Perform hypothesis test
        test_results = analyzer.hypothesis_testing(
            request.column1, 
            request.column2, 
            request.test_type.value
        )
        
        # Generate recommendations
        recommendations = []
        if test_results['is_significant']:
            recommendations.append("The test result is statistically significant (p < 0.05)")
            if test_results['test_type'] == 'Pearson Correlation':
                recommendations.append("Consider investigating the relationship further with regression analysis")
            elif test_results['test_type'] == 'One-way ANOVA':
                recommendations.append("Consider post-hoc tests to identify which groups differ significantly")
            elif test_results['test_type'] == 'Chi-square Test of Independence':
                recommendations.append("Examine the contingency table to understand the nature of the association")
        else:
            recommendations.append("The test result is not statistically significant (p >= 0.05)")
            recommendations.append("There is insufficient evidence to reject the null hypothesis")
        
        return HypothesisTestResponse(
            status=StatusEnum.SUCCESS,
            message=f"Hypothesis test completed: {test_results['test_type']}",
            test_result=test_results,
            recommendations=recommendations
        )
        
    except AnalysisError as e:
        logger.error(f"Hypothesis test error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error performing hypothesis test: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to perform hypothesis test")


@router.post("/categorical", response_model=CategoricalAnalysisResponse)
async def analyze_categorical(
    request: CategoricalAnalysisRequest,
    analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)
):
    """
    Analyze categorical variables.
    
    If no column is specified, analyzes all categorical columns.
    """
    try:
        logger.info(f"Analyzing categorical data for column: {request.column}")
        
        # Validate column if specified
        if request.column:
            if request.column not in analyzer.categorical_columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{request.column}' is not categorical. Available categorical columns: {analyzer.categorical_columns}"
                )
        
        # Analyze categorical data
        categorical_results = analyzer.categorical_analysis(request.column)
        
        # Generate summary
        if request.column:
            col_results = categorical_results[request.column]
            summary = f"Categorical analysis for '{request.column}': "
            summary += f"{col_results['unique_values']} unique values, "
            summary += f"most frequent: '{col_results['most_frequent']['value']}' "
            summary += f"({col_results['most_frequent']['percentage']:.1f}%)"
        else:
            summary = f"Categorical analysis completed for {len(categorical_results)} columns. "
            total_categories = sum(results['unique_values'] for results in categorical_results.values())
            summary += f"Total unique categories across all columns: {total_categories}"
        
        return CategoricalAnalysisResponse(
            status=StatusEnum.SUCCESS,
            message="Categorical analysis completed successfully",
            analysis=categorical_results,
            summary=summary
        )
        
    except AnalysisError as e:
        logger.error(f"Categorical analysis error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error analyzing categorical data: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to analyze categorical data")


@router.get("/summary")
async def get_analysis_summary(analyzer: StatisticalAnalyzer = Depends(get_statistical_analyzer)):
    """
    Get a comprehensive analysis summary of the dataset.
    """
    try:
        logger.info("Generating comprehensive analysis summary")
        
        # Get basic info
        basic_info = analyzer.get_basic_info()
        
        # Get schema overview
        schema_overview = analyzer.get_schema_overview()
        
        # Get variable summaries
        variable_summaries = analyzer.get_variable_summaries(limit=10)
        
        # Quick correlation analysis if we have numeric columns
        correlation_summary = None
        if len(analyzer.numeric_columns) >= 2:
            try:
                correlation_results = analyzer.correlation_analysis()
                correlation_summary = {
                    "strongest_correlation": correlation_results['strongest_correlations'][0] if correlation_results['strongest_correlations'] else None,
                    "strong_correlations_count": correlation_results['summary']['strong_correlations']
                }
            except Exception:
                pass
        
        # Quick outlier summary
        outlier_summary = None
        if analyzer.numeric_columns:
            try:
                outlier_results = analyzer.outlier_detection(analyzer.numeric_columns[0], ["iqr"])
                first_col = analyzer.numeric_columns[0]
                outlier_summary = {
                    "sample_column": first_col,
                    "outlier_count": outlier_results[first_col]['iqr']['count'],
                    "outlier_percentage": outlier_results[first_col]['iqr']['percentage']
                }
            except Exception:
                pass
        
        return {
            "status": "success",
            "message": "Analysis summary generated successfully",
            "summary": {
                "basic_info": basic_info,
                "schema_overview": schema_overview,
                "variable_summaries": variable_summaries,
                "correlation_summary": correlation_summary,
                "outlier_summary": outlier_summary,
                "recommendations": [
                    "Start with descriptive statistics to understand your data",
                    "Check for correlations between numeric variables",
                    "Identify and investigate outliers",
                    "Analyze distributions for normality assumptions",
                    "Use hypothesis testing to validate relationships"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating analysis summary: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate analysis summary")

