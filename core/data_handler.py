"""Data handling and processing module for the Statistical Analysis Bot API."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from fastapi import UploadFile
import io

from utils.exceptions import DataProcessingError, FileUploadError, ValidationError
from utils.logging import LoggerMixin


class DataHandler(LoggerMixin):
    """Handles data loading, processing, and validation."""
    
    def __init__(self):
        self.supported_extensions = ['.csv', '.xlsx', '.xls']
        self.max_file_size = 100 * 1024 * 1024  # 100MB
    
    async def process_uploaded_files(self, files: List[UploadFile]) -> pd.DataFrame:
        """Process uploaded files and return combined DataFrame."""
        self.log_operation("process_uploaded_files", file_count=len(files))
        
        if not files:
            raise FileUploadError("No files provided")
        
        all_dataframes = []
        
        for file in files:
            try:
                # Validate file
                self._validate_file(file)
                
                # Read file content
                content = await file.read()
                df = self._read_file_content(content, file.filename)
                
                if df.empty:
                    self.logger.warning(f"No data found in file: {file.filename}")
                    continue
                
                # Add source file identifier
                df['__source_file__'] = file.filename
                all_dataframes.append(df)
                
                self.logger.info(f"Processed file: {file.filename} ({len(df)} rows, {len(df.columns)} columns)")
                
            except Exception as e:
                self.logger.error(f"Error processing file {file.filename}: {str(e)}")
                raise DataProcessingError(f"Failed to process file {file.filename}: {str(e)}")
        
        if not all_dataframes:
            raise DataProcessingError("No valid data found in any uploaded files")
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        # Optimize data types
        optimized_df = self.optimize_data_types(combined_df)
        
        self.logger.info(f"Combined dataset: {len(optimized_df)} rows, {len(optimized_df.columns)} columns")
        return optimized_df
    
    def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file."""
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.supported_extensions:
            raise FileUploadError(
                f"Unsupported file type: {file_ext}. Supported types: {', '.join(self.supported_extensions)}"
            )
        
        # Check file size (if available)
        if hasattr(file, 'size') and file.size and file.size > self.max_file_size:
            raise FileUploadError(f"File too large: {file.size} bytes. Maximum allowed: {self.max_file_size} bytes")
    
    def _read_file_content(self, content: bytes, filename: str) -> pd.DataFrame:
        """Read file content and return DataFrame."""
        file_ext = Path(filename).suffix.lower()
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(io.BytesIO(content), low_memory=False)
            
            elif file_ext in ['.xlsx', '.xls']:
                # Handle Excel files with multiple sheets
                excel_file = pd.ExcelFile(io.BytesIO(content))
                dataframes = []
                
                for sheet_name in excel_file.sheet_names:
                    sheet_df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    # Add sheet identifier if multiple sheets
                    if len(excel_file.sheet_names) > 1:
                        sheet_df['__source_sheet__'] = sheet_name
                    
                    dataframes.append(sheet_df)
                
                return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
            
            else:
                raise DataProcessingError(f"Unsupported file extension: {file_ext}")
                
        except Exception as e:
            raise DataProcessingError(f"Failed to read file content: {str(e)}")
    
    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for statistical analysis."""
        self.log_operation("optimize_data_types", columns=len(df.columns))
        
        optimized_df = df.copy()
        
        for column in optimized_df.columns:
            try:
                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(optimized_df[column]):
                    continue
                
                # Try to convert to numeric
                numeric_series = pd.to_numeric(optimized_df[column], errors='coerce')
                
                # If conversion successful for most values, use numeric type
                non_null_count = numeric_series.notna().sum()
                total_count = len(numeric_series)
                
                if non_null_count > 0 and (non_null_count / total_count) > 0.5:
                    # Determine best numeric type
                    if numeric_series.notna().all():  # No NaN values
                        if (numeric_series % 1 == 0).all():  # All integers
                            # Use appropriate integer type
                            min_val, max_val = numeric_series.min(), numeric_series.max()
                            if min_val >= -128 and max_val <= 127:
                                optimized_df[column] = numeric_series.astype('int8')
                            elif min_val >= -32768 and max_val <= 32767:
                                optimized_df[column] = numeric_series.astype('int16')
                            elif min_val >= -2147483648 and max_val <= 2147483647:
                                optimized_df[column] = numeric_series.astype('int32')
                            else:
                                optimized_df[column] = numeric_series.astype('int64')
                        else:
                            optimized_df[column] = numeric_series.astype('float32')
                    else:
                        # Has NaN values, use nullable types
                        if (numeric_series.dropna() % 1 == 0).all():
                            optimized_df[column] = numeric_series.astype('Int64')
                        else:
                            optimized_df[column] = numeric_series.astype('float64')
                    
                    self.logger.debug(f"Converted {column} to numeric type")
                else:
                    # Keep as categorical or string
                    unique_ratio = optimized_df[column].nunique() / len(optimized_df[column])
                    if unique_ratio < 0.5:  # Low cardinality
                        optimized_df[column] = optimized_df[column].astype('category')
                        self.logger.debug(f"Converted {column} to categorical type")
                    else:
                        optimized_df[column] = optimized_df[column].astype('string')
                        
            except Exception as e:
                self.logger.warning(f"Could not optimize column {column}: {str(e)}")
                continue
        
        return optimized_df
    
    def get_column_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive column information."""
        self.log_operation("get_column_info", columns=len(df.columns))
        
        info = {
            'total_columns': len(df.columns),
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'text_columns': [],
            'column_types': {},
            'missing_values': {},
            'unique_counts': {}
        }
        
        for column in df.columns:
            col_type = str(df[column].dtype)
            info['column_types'][column] = col_type
            info['missing_values'][column] = int(df[column].isnull().sum())
            info['unique_counts'][column] = int(df[column].nunique())
            
            if pd.api.types.is_numeric_dtype(df[column]):
                info['numeric_columns'].append(column)
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                info['datetime_columns'].append(column)
            elif pd.api.types.is_categorical_dtype(df[column]) or df[column].nunique() / len(df) < 0.5:
                info['categorical_columns'].append(column)
            else:
                info['text_columns'].append(column)
        
        return info
    
    def get_column_unique_values(self, df: pd.DataFrame, column: str, limit: int = 100) -> List[Any]:
        """Get unique values from a column with a limit."""
        if column not in df.columns:
            raise ValidationError(f"Column '{column}' not found in dataset")
        
        unique_values = df[column].dropna().unique()
        
        if len(unique_values) > limit:
            self.logger.warning(f"Column {column} has {len(unique_values)} unique values, returning first {limit}")
            return unique_values[:limit].tolist()
        
        return unique_values.tolist()
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return quality metrics."""
        self.log_operation("validate_data_quality", rows=len(df), columns=len(df.columns))
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            'duplicate_rows': int(df.duplicated().sum()),
            'columns_with_missing_data': df.columns[df.isnull().any()].tolist(),
            'columns_all_missing': df.columns[df.isnull().all()].tolist(),
            'numeric_columns_count': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns_count': len(df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        return quality_report
    
    def prepare_for_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for statistical analysis."""
        self.log_operation("prepare_for_analysis", original_shape=df.shape)
        
        # Remove completely empty rows and columns
        cleaned_df = df.dropna(how='all').dropna(axis=1, how='all')
        
        rows_removed = len(df) - len(cleaned_df)
        cols_removed = len(df.columns) - len(cleaned_df.columns)
        
        if rows_removed > 0:
            self.logger.info(f"Removed {rows_removed} completely empty rows")
        if cols_removed > 0:
            self.logger.info(f"Removed {cols_removed} completely empty columns")
        
        return cleaned_df

