# PandasAI Integration Guide

## Overview

The FastAPI Statistical Analysis Bot now includes **PandasAI integration** for enhanced natural language data interaction. This allows users to ask questions about their data in plain English and get intelligent responses powered by AI.

## What's New

### 🤖 PandasAI Integration
- **Natural Language Queries**: Ask questions like "What is the average salary?" or "Show me departments and their average salaries"
- **Intelligent Data Analysis**: PandasAI automatically generates and executes appropriate code to answer your questions
- **Multiple Response Types**: Handles numeric results, DataFrames, strings, and complex analysis
- **SQL Generation**: Automatically generates SQL queries for complex aggregations and joins

### 🔧 Technical Implementation

#### Core Components
1. **Custom OpenAI LLM Wrapper**: Integrates with PandasAI's LLM interface
2. **SmartDataframe**: Enhanced DataFrame with AI capabilities
3. **Response Processing**: Intelligent handling of different response types
4. **Fallback Logic**: Graceful handling when PandasAI can't process queries

#### Key Features
- **Model Compatibility**: Uses supported models (gpt-4.1-mini, gpt-4.1-nano, gemini-2.5-flash)
- **Error Handling**: Comprehensive error handling with fallback responses
- **Conversation History**: Maintains context across multiple queries
- **Streaming Support**: Real-time response generation

## Usage Examples

### Basic Queries
```bash
# Simple statistics
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the average salary?"}'

# Response: {"response": "Result: 59400.0"}
```

### Complex Analysis
```bash
# Grouped analysis
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me the departments and their average salaries"}'

# Response: 
# {
#   "response": "Result:     department  average_salary\n0  Engineering    61000.000000\n1    Marketing    59666.666667\n2        Sales    55000.000000"
# }
```

### Supported Query Types
- **Descriptive Statistics**: "What is the mean/median/mode of column X?"
- **Aggregations**: "Sum/average/count by department"
- **Comparisons**: "Compare salaries between departments"
- **Filtering**: "Show me employees with salary > 60000"
- **Correlations**: "Is there a correlation between age and salary?"
- **Data Exploration**: "Describe the dataset", "Show me unique values"

## Configuration

### Environment Variables
```bash
# Required for PandasAI functionality
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Use specific model
OPENAI_MODEL=gpt-4.1-mini  # Default model used
```

### Supported Models
- `gpt-4.1-mini` (default)
- `gpt-4.1-nano`
- `gemini-2.5-flash`

## API Endpoints

### Chat Query (Enhanced)
```
POST /api/chat/query
```

**Request:**
```json
{
  "query": "Your natural language question",
  "include_history": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Query processed successfully",
  "response": "AI-generated response",
  "conversation_history": [...],
  "suggested_questions": [...],
  "analysis_performed": ["descriptive_statistics"]
}
```

### Streaming Chat
```
POST /api/chat/query/stream
```
Real-time streaming responses for better user experience.

## Technical Details

### PandasAI Workflow
1. **Query Processing**: User query is sent to PandasAI
2. **Code Generation**: PandasAI generates appropriate Python/SQL code
3. **Execution**: Code is executed against the dataset
4. **Response Formatting**: Results are formatted for API response
5. **Fallback Handling**: If PandasAI fails, fallback to statistical analyzer

### Response Types Handled
- **Numeric Values**: Direct numerical answers
- **DataFrames**: Tabular results formatted as strings
- **Strings**: Text-based responses
- **Boolean Values**: True/false answers
- **None/Empty**: Fallback to statistical analysis

### Error Handling
- **Model Errors**: Automatic model fallback
- **API Errors**: Graceful error messages
- **Processing Errors**: Fallback to built-in statistical methods
- **Timeout Handling**: Prevents hanging requests

## Performance Considerations

### Optimization Features
- **Caching Disabled**: Ensures fresh results for each query
- **Memory Management**: Efficient handling of large datasets
- **Response Streaming**: Reduces perceived latency
- **Fallback Logic**: Ensures responses even when AI fails

### Limitations
- **Model Dependencies**: Requires valid OpenAI API key
- **Query Complexity**: Very complex queries may need refinement
- **Data Size**: Performance depends on dataset size
- **Rate Limits**: Subject to OpenAI API rate limits

## Migration from Previous Version

### What Changed
1. **Enhanced LLM Agent**: Now uses PandasAI for query processing
2. **Improved Responses**: More intelligent and context-aware answers
3. **Better Error Handling**: Graceful fallbacks when AI processing fails
4. **Updated Dependencies**: Added PandasAI and updated OpenAI client

### Backward Compatibility
- All existing endpoints remain functional
- Statistical analysis endpoints unchanged
- API response format maintained
- Configuration options preserved

## Troubleshooting

### Common Issues

#### 1. Model Not Supported Error
```
Error: Unsupported model. Only the following models are allowed: gpt-4.1-mini, gpt-4.1-nano, gemini-2.5-flash
```
**Solution**: Update the model in the LLM configuration to use a supported model.

#### 2. OpenAI API Key Issues
```
Error: OpenAI API key not configured
```
**Solution**: Ensure `OPENAI_API_KEY` is set in your environment variables.

#### 3. PandasAI Processing Errors
**Symptoms**: Queries return fallback responses
**Solution**: Check query phrasing, ensure data compatibility, verify API connectivity.

### Debug Mode
Enable debug logging to see PandasAI's internal processing:
```bash
LOG_LEVEL=DEBUG python -m app.main
```

## Examples and Use Cases

### Business Intelligence
```bash
# Revenue analysis
"What is the total salary cost by department?"

# Performance metrics
"Show me the top 5 employees by salary"

# Trend analysis
"Compare average experience by department"
```

### Data Exploration
```bash
# Data overview
"Describe this dataset"

# Missing data
"Are there any missing values?"

# Data quality
"Show me duplicate records"
```

### Statistical Analysis
```bash
# Correlations
"Is there a correlation between age and salary?"

# Distributions
"What is the distribution of salaries?"

# Outliers
"Find outliers in the salary column"
```

## Future Enhancements

### Planned Features
- **Chart Generation**: Visual responses for data visualization
- **Advanced Analytics**: Time series analysis, forecasting
- **Custom Functions**: Domain-specific analysis functions
- **Multi-table Queries**: Cross-dataset analysis
- **Export Capabilities**: Download results in various formats

### Integration Opportunities
- **Business Intelligence Tools**: Connect to BI dashboards
- **Reporting Systems**: Automated report generation
- **Data Pipelines**: Integration with ETL processes
- **Machine Learning**: Predictive analytics capabilities

## Support

For issues related to PandasAI integration:
1. Check the logs for detailed error messages
2. Verify OpenAI API key and model compatibility
3. Test with simple queries first
4. Review the PandasAI documentation for advanced usage

The integration maintains full backward compatibility while adding powerful AI-driven data interaction capabilities to your statistical analysis workflow.

