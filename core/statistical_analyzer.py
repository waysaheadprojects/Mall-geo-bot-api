"""Statistical analysis engine for the FastAPI Statistical Analysis Bot."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Union
import warnings

from utils.exceptions import AnalysisError, ValidationError
from utils.logging import LoggerMixin

warnings.filterwarnings('ignore', category=RuntimeWarning)


class StatisticalAnalyzer(LoggerMixin):
    """Comprehensive statistical analysis engine."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_columns = self._get_numeric_columns()
        self.categorical_columns = self._get_categorical_columns()
        self.log_operation("initialized", 
                          rows=len(df), 
                          columns=len(df.columns),
                          numeric_cols=len(self.numeric_columns),
                          categorical_cols=len(self.categorical_columns))
    
    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns."""
        numeric_cols = []
        for col in self.df.columns:
            try:
                pd.to_numeric(self.df[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        return numeric_cols
    
    def _get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns."""
        return [col for col in self.df.columns if col not in self.numeric_columns]
    
    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic dataset information."""
        self.log_operation("get_basic_info")
        
        info = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "numeric_columns": len(self.numeric_columns),
            "categorical_columns": len(self.categorical_columns),
            "missing_values": int(self.df.isnull().sum().sum()),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / 1024 / 1024),
            "duplicate_rows": int(self.df.duplicated().sum())
        }
        
        return info
    
    def get_schema_overview(self) -> Dict[str, Any]:
        """Get comprehensive schema overview."""
        self.log_operation("get_schema_overview")
        
        total_rows = len(self.df)
        schema_info = []
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / total_rows) * 100
            unique_count = self.df[col].nunique()
            unique_pct = (unique_count / total_rows) * 100
            
            schema_info.append({
                "column": col,
                "dtype": dtype,
                "null_count": int(null_count),
                "null_percentage": float(null_pct),
                "unique_count": int(unique_count),
                "unique_percentage": float(unique_pct),
                "is_numeric": col in self.numeric_columns
            })
        
        return {
            "columns": schema_info,
            "summary": {
                "total_columns": len(self.df.columns),
                "numeric_columns": self.numeric_columns,
                "categorical_columns": self.categorical_columns
            }
        }
    
    def get_variable_summaries(self, limit: int = 5) -> Dict[str, Any]:
        """Get summary statistics for numeric variables."""
        self.log_operation("get_variable_summaries", limit=limit)
        
        summaries = {}
        columns_to_analyze = self.numeric_columns[:limit]
        
        for col in columns_to_analyze:
            try:
                data = pd.to_numeric(self.df[col], errors='coerce')
                summaries[col] = {
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "missing": int(data.isnull().sum()),
                    "count": int(data.count())
                }
            except Exception as e:
                self.logger.warning(f"Could not summarize column {col}: {str(e)}")
                continue
        
        return summaries
    
    def descriptive_statistics(self, column: Optional[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive descriptive statistics."""
        self.log_operation("descriptive_statistics", column=column)
        
        if column:
            if column not in self.df.columns:
                raise ValidationError(f"Column '{column}' not found in dataset")
            if column not in self.numeric_columns:
                raise ValidationError(f"Column '{column}' is not numeric")
            
            data = pd.to_numeric(self.df[column], errors='coerce').dropna()
            return {column: self._calculate_single_column_stats(data, column)}
        
        # Calculate for all numeric columns
        results = {}
        for col in self.numeric_columns:
            try:
                data = pd.to_numeric(self.df[col], errors='coerce').dropna()
                results[col] = self._calculate_single_column_stats(data, col)
            except Exception as e:
                self.logger.warning(f"Could not analyze column {col}: {str(e)}")
                continue
        
        return results
    
    def _calculate_single_column_stats(self, data: pd.Series, column_name: str) -> Dict[str, Any]:
        """Calculate statistics for a single column."""
        try:
            mode_val = data.mode()
            mode_result = float(mode_val.iloc[0]) if not mode_val.empty else None
            
            stats_dict = {
                "count": int(len(data)),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "mode": mode_result,
                "std_dev": float(data.std()),
                "variance": float(data.var()),
                "min": float(data.min()),
                "max": float(data.max()),
                "range": float(data.max() - data.min()),
                "q1": float(data.quantile(0.25)),
                "q3": float(data.quantile(0.75)),
                "iqr": float(data.quantile(0.75) - data.quantile(0.25)),
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data))
            }
            
            # Coefficient of variation
            if data.mean() != 0:
                stats_dict["coef_var"] = float((data.std() / data.mean()) * 100)
            else:
                stats_dict["coef_var"] = None
            
            return stats_dict
            
        except Exception as e:
            raise AnalysisError(f"Failed to calculate statistics for column {column_name}: {str(e)}")
    
    def correlation_analysis(self, method: str = "pearson") -> Dict[str, Any]:
        """Perform correlation analysis."""
        self.log_operation("correlation_analysis", method=method)
        
        if len(self.numeric_columns) < 2:
            raise AnalysisError("Need at least 2 numeric columns for correlation analysis")
        
        try:
            df_numeric = self.df[self.numeric_columns].apply(pd.to_numeric, errors='coerce')
            correlation_matrix = df_numeric.corr(method=method)
            
            # Extract correlation pairs
            pairs = []
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns[i+1:], start=i+1):
                    corr_val = correlation_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        pairs.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr_val),
                            "strength": self._interpret_correlation(abs(corr_val)),
                            "direction": "positive" if corr_val > 0 else "negative"
                        })
            
            # Sort by absolute correlation value
            pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return {
                "method": method,
                "correlation_matrix": correlation_matrix.to_dict(),
                "strongest_correlations": pairs[:10],
                "interpretation": self._interpret_correlation_matrix(pairs),
                "summary": {
                    "total_pairs": len(pairs),
                    "strong_correlations": len([p for p in pairs if abs(p["correlation"]) >= 0.6]),
                    "moderate_correlations": len([p for p in pairs if 0.4 <= abs(p["correlation"]) < 0.6])
                }
            }
            
        except Exception as e:
            raise AnalysisError(f"Correlation analysis failed: {str(e)}")
    
    def _interpret_correlation(self, abs_corr: float) -> str:
        """Interpret correlation strength."""
        if abs_corr >= 0.8:
            return "Very Strong"
        elif abs_corr >= 0.6:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def _interpret_correlation_matrix(self, correlations: List[Dict]) -> str:
        """Provide interpretation of correlation results."""
        if not correlations:
            return "No significant correlations found."
        
        strongest = correlations[0]
        return (f"Strongest correlation is between {strongest['column1']} and {strongest['column2']} "
                f"(r = {strongest['correlation']:.3f}, {strongest['strength']} {strongest['direction']} correlation).")
    
    def outlier_detection(self, column: Optional[str] = None, methods: List[str] = None) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        self.log_operation("outlier_detection", column=column, methods=methods)
        
        if methods is None:
            methods = ["iqr", "zscore"]
        
        if column:
            if column not in self.numeric_columns:
                raise ValidationError(f"Column '{column}' is not numeric")
            return {column: self._detect_outliers_single_column(column, methods)}
        
        # Analyze all numeric columns
        results = {}
        for col in self.numeric_columns:
            try:
                results[col] = self._detect_outliers_single_column(col, methods)
            except Exception as e:
                self.logger.warning(f"Could not detect outliers for column {col}: {str(e)}")
                continue
        
        return results
    
    def _detect_outliers_single_column(self, column: str, methods: List[str]) -> Dict[str, Any]:
        """Detect outliers for a single column."""
        data = pd.to_numeric(self.df[column], errors='coerce').dropna()
        results = {}
        
        if "iqr" in methods:
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((data < lower_bound) | (data > upper_bound))
            
            results["iqr"] = {
                "count": int(outliers.sum()),
                "percentage": float((outliers.sum() / len(data)) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_values": data[outliers].tolist()[:10]  # Limit to first 10
            }
        
        if "zscore" in methods:
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > 3
            
            results["zscore"] = {
                "count": int(outliers.sum()),
                "percentage": float((outliers.sum() / len(data)) * 100),
                "threshold": 3.0,
                "outlier_values": data[outliers].tolist()[:10]  # Limit to first 10
            }
        
        return results
    
    def distribution_analysis(self, column: str) -> Dict[str, Any]:
        """Analyze distribution characteristics of a column."""
        self.log_operation("distribution_analysis", column=column)
        
        if column not in self.numeric_columns:
            raise ValidationError(f"Column '{column}' is not numeric")
        
        try:
            data = pd.to_numeric(self.df[column], errors='coerce').dropna()
            
            # Normality test (use sample if data is large)
            sample_size = min(5000, len(data))
            sample_data = data.sample(sample_size) if len(data) > sample_size else data
            
            shapiro_stat, shapiro_p = stats.shapiro(sample_data)
            
            # Distribution shape metrics
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            return {
                "column": column,
                "sample_size": len(data),
                "normality_test": {
                    "test": "Shapiro-Wilk",
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": bool(shapiro_p > 0.05),
                    "sample_size_used": sample_size
                },
                "shape_metrics": {
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "interpretation": self._interpret_distribution_shape(skewness, kurtosis)
                },
                "distribution_type": self._suggest_distribution_type(data)
            }
            
        except Exception as e:
            raise AnalysisError(f"Distribution analysis failed for column {column}: {str(e)}")
    
    def _interpret_distribution_shape(self, skewness: float, kurtosis: float) -> Dict[str, str]:
        """Interpret distribution shape characteristics."""
        if abs(skewness) < 0.5:
            skew_desc = "approximately symmetric"
        elif skewness > 0:
            skew_desc = "right-skewed (positive skew)"
        else:
            skew_desc = "left-skewed (negative skew)"
        
        if abs(kurtosis) < 0.5:
            kurt_desc = "normal tail behavior (mesokurtic)"
        elif kurtosis > 0:
            kurt_desc = "heavy tails (leptokurtic)"
        else:
            kurt_desc = "light tails (platykurtic)"
        
        return {
            "skewness_interpretation": skew_desc,
            "kurtosis_interpretation": kurt_desc
        }
    
    def _suggest_distribution_type(self, data: pd.Series) -> str:
        """Suggest likely distribution type based on data characteristics."""
        skewness = abs(stats.skew(data))
        kurtosis = stats.kurtosis(data)
        
        if skewness < 0.5 and abs(kurtosis) < 0.5:
            return "Normal distribution"
        elif skewness > 1:
            return "Exponential or Log-normal distribution"
        elif data.min() >= 0 and skewness > 0.5:
            return "Gamma or Weibull distribution"
        else:
            return "Non-standard distribution"
    
    def categorical_analysis(self, column: Optional[str] = None) -> Dict[str, Any]:
        """Analyze categorical data."""
        self.log_operation("categorical_analysis", column=column)
        
        if column:
            if column not in self.categorical_columns:
                raise ValidationError(f"Column '{column}' is not categorical")
            return {column: self._analyze_single_categorical(column)}
        
        # Analyze all categorical columns
        results = {}
        for col in self.categorical_columns:
            try:
                results[col] = self._analyze_single_categorical(col)
            except Exception as e:
                self.logger.warning(f"Could not analyze categorical column {col}: {str(e)}")
                continue
        
        return results
    
    def _analyze_single_categorical(self, column: str) -> Dict[str, Any]:
        """Analyze a single categorical column."""
        data = self.df[column].dropna()
        value_counts = data.value_counts()
        
        return {
            "unique_values": int(value_counts.size),
            "most_frequent": {
                "value": str(value_counts.index[0]),
                "count": int(value_counts.iloc[0]),
                "percentage": float((value_counts.iloc[0] / len(data)) * 100)
            },
            "distribution": {
                str(k): int(v) for k, v in value_counts.head(10).items()
            },
            "entropy": float(stats.entropy(value_counts.values)),
            "missing_values": int(self.df[column].isnull().sum())
        }
    
    def hypothesis_testing(self, column1: str, column2: str, test_type: str = "auto") -> Dict[str, Any]:
        """Perform hypothesis testing between two variables."""
        self.log_operation("hypothesis_testing", col1=column1, col2=column2, test_type=test_type)
        
        if column1 not in self.df.columns or column2 not in self.df.columns:
            raise ValidationError("One or both columns not found in dataset")
        
        try:
            if test_type == "auto":
                # Determine appropriate test based on variable types
                if column1 in self.numeric_columns and column2 in self.numeric_columns:
                    return self._correlation_test(column1, column2)
                elif column1 in self.numeric_columns and column2 in self.categorical_columns:
                    return self._anova_test(column1, column2)
                elif column1 in self.categorical_columns and column2 in self.numeric_columns:
                    return self._anova_test(column2, column1)
                else:
                    return self._chi_square_test(column1, column2)
            else:
                raise ValidationError(f"Unsupported test type: {test_type}")
                
        except Exception as e:
            raise AnalysisError(f"Hypothesis testing failed: {str(e)}")
    
    def _correlation_test(self, col1: str, col2: str) -> Dict[str, Any]:
        """Perform correlation test between two numeric variables."""
        data1 = pd.to_numeric(self.df[col1], errors='coerce')
        data2 = pd.to_numeric(self.df[col2], errors='coerce')
        
        # Use only rows where both values are not null
        valid_idx = data1.dropna().index.intersection(data2.dropna().index)
        
        if len(valid_idx) < 3:
            raise AnalysisError("Insufficient valid data points for correlation test")
        
        correlation, p_value = stats.pearsonr(data1.loc[valid_idx], data2.loc[valid_idx])
        
        return {
            "test_type": "Pearson Correlation",
            "variables": [col1, col2],
            "correlation": float(correlation),
            "p_value": float(p_value),
            "is_significant": bool(p_value < 0.05),
            "sample_size": len(valid_idx),
            "interpretation": f"{'Significant' if p_value < 0.05 else 'Non-significant'} {self._interpret_correlation(abs(correlation)).lower()} correlation"
        }
    
    def _anova_test(self, numeric_col: str, categorical_col: str) -> Dict[str, Any]:
        """Perform ANOVA test between numeric and categorical variables."""
        numeric_data = pd.to_numeric(self.df[numeric_col], errors='coerce')
        categorical_data = self.df[categorical_col]
        
        # Create groups
        groups = []
        group_names = []
        
        for category in categorical_data.dropna().unique():
            group_data = numeric_data[categorical_data == category].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(str(category))
        
        if len(groups) < 2:
            raise AnalysisError("Need at least 2 groups for ANOVA test")
        
        f_statistic, p_value = stats.f_oneway(*groups)
        
        return {
            "test_type": "One-way ANOVA",
            "variables": [numeric_col, categorical_col],
            "f_statistic": float(f_statistic),
            "p_value": float(p_value),
            "is_significant": bool(p_value < 0.05),
            "groups": group_names,
            "group_sizes": [len(group) for group in groups],
            "interpretation": f"{'Significant' if p_value < 0.05 else 'Non-significant'} difference between groups"
        }
    
    def _chi_square_test(self, col1: str, col2: str) -> Dict[str, Any]:
        """Perform chi-square test between two categorical variables."""
        contingency_table = pd.crosstab(self.df[col1], self.df[col2])
        
        if contingency_table.size == 0:
            raise AnalysisError("No valid data for chi-square test")
        
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            "test_type": "Chi-square Test of Independence",
            "variables": [col1, col2],
            "chi2_statistic": float(chi2_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "is_significant": bool(p_value < 0.05),
            "contingency_table": contingency_table.to_dict(),
            "interpretation": f"{'Significant' if p_value < 0.05 else 'Non-significant'} association between variables"
        }

