# Statistical Analysis Bot API

A production-ready FastAPI application for statistical analysis with LLM-powered natural language queries. This API provides comprehensive statistical analysis tools and allows users to interact with their data using natural language.

## Features

### 🔢 Statistical Analysis
- **Descriptive Statistics**: Mean, median, mode, standard deviation, quartiles, skewness, kurtosis
- **Correlation Analysis**: Pearson, Spearman, and Kendall correlation with interpretation
- **Outlier Detection**: IQR and Z-score methods with visualization support
- **Distribution Analysis**: Normality tests, distribution shape analysis, and type suggestions
- **Hypothesis Testing**: Automatic test selection (correlation, ANOVA, chi-square) based on variable types
- **Categorical Analysis**: Frequency distributions, entropy measures, and category insights

### 🤖 LLM Integration
- **Natural Language Queries**: Ask questions about your data in plain English
- **Streaming Responses**: Real-time response generation for better user experience
- **Context-Aware Analysis**: AI understands your dataset structure and provides relevant insights
- **Suggested Questions**: Get recommendations for further analysis based on your data
- **Conversation History**: Maintain context across multiple queries

### 📊 Data Management
- **File Upload**: Support for CSV and Excel files with multiple sheets
- **Data Optimization**: Automatic data type optimization for better performance
- **Quality Assessment**: Comprehensive data quality reports with missing value analysis
- **Persistent Storage**: Automatic data persistence with metadata tracking
- **Schema Analysis**: Detailed column information and data type detection

### 🚀 Production Features
- **Comprehensive Error Handling**: Detailed error messages with proper HTTP status codes
- **Request Validation**: Pydantic models for request/response validation
- **Logging**: Structured logging with configurable levels
- **CORS Support**: Cross-origin resource sharing for frontend integration
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation
- **Health Checks**: Service health monitoring endpoints

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (for LLM features)

### Installation

1. **Clone or extract the project**
```bash
cd fastapi_statistical_bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env file with your OpenAI API key and other settings
```

4. **Run the application**
```bash
python -m app.main
```

The API will be available at `http://localhost:8000`

### Using Docker (Optional)

```bash
# Build the image
docker build -t statistical-bot-api .

# Run the container
docker run -p 8000:8000 --env-file .env statistical-bot-api
```

## API Usage

### 1. Upload Data
```bash
curl -X POST "http://localhost:8000/api/data/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@your_data.csv"
```

### 2. Get Data Information
```bash
curl -X GET "http://localhost:8000/api/data/info"
```

### 3. Perform Statistical Analysis
```bash
# Descriptive statistics
curl -X POST "http://localhost:8000/api/analysis/descriptive" \
  -H "Content-Type: application/json" \
  -d '{"column": "your_column_name"}'

# Correlation analysis
curl -X POST "http://localhost:8000/api/analysis/correlation" \
  -H "Content-Type: application/json" \
  -d '{"method": "pearson"}'
```

### 4. Ask Natural Language Questions
```bash
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main statistics for my numeric columns?"}'
```

## API Endpoints

### Data Management
- `POST /api/data/upload` - Upload and process data files
- `GET /api/data/info` - Get dataset information
- `GET /api/data/columns` - List all columns with details
- `GET /api/data/columns/{column_name}` - Get specific column information
- `DELETE /api/data/clear` - Clear stored data
- `GET /api/data/status` - Get data loading status

### Statistical Analysis
- `POST /api/analysis/descriptive` - Calculate descriptive statistics
- `POST /api/analysis/correlation` - Perform correlation analysis
- `POST /api/analysis/outliers` - Detect outliers
- `POST /api/analysis/distribution` - Analyze distributions
- `POST /api/analysis/hypothesis` - Perform hypothesis testing
- `POST /api/analysis/categorical` - Analyze categorical variables
- `GET /api/analysis/summary` - Get comprehensive analysis summary

### Chat & LLM
- `POST /api/chat/query` - Process natural language queries
- `POST /api/chat/query/stream` - Stream query responses
- `GET /api/chat/history` - Get conversation history
- `DELETE /api/chat/history` - Clear conversation history
- `GET /api/chat/suggestions` - Get suggested questions
- `GET /api/chat/context` - Get dataset context for chat

### Health & Status
- `GET /health` - Health check
- `GET /` - API information

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM features | Required |
| `OPENAI_API_BASE` | OpenAI API base URL | https://api.openai.com/v1 |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `DEBUG` | Enable debug mode | false |
| `LOG_LEVEL` | Logging level | INFO |
| `DATA_STORAGE_DIR` | Data storage directory | data_storage |
| `MAX_FILE_SIZE` | Maximum file size in bytes | 104857600 (100MB) |

### CORS Configuration
The API supports CORS for frontend integration. Configure allowed origins in the environment variables:

```bash
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
```

## Data Types Supported

### File Formats
- **CSV**: Comma-separated values with automatic delimiter detection
- **Excel**: .xlsx and .xls files with multiple sheet support

### Data Types
- **Numeric**: Integers and floating-point numbers with automatic optimization
- **Categorical**: Text and categorical data with cardinality analysis
- **DateTime**: Date and time data with automatic parsing
- **Boolean**: True/false values

## Statistical Methods

### Descriptive Statistics
- Central tendency: mean, median, mode
- Variability: standard deviation, variance, range, IQR
- Shape: skewness, kurtosis
- Position: quartiles, percentiles

### Correlation Analysis
- **Pearson**: Linear relationships between continuous variables
- **Spearman**: Monotonic relationships, robust to outliers
- **Kendall**: Rank-based correlation for small samples

### Outlier Detection
- **IQR Method**: Interquartile range with 1.5×IQR rule
- **Z-Score Method**: Standard deviation-based with 3-sigma rule

### Hypothesis Testing
- **Correlation Test**: Pearson correlation significance
- **ANOVA**: Compare means across groups
- **Chi-Square**: Test independence between categorical variables

## Error Handling

The API provides comprehensive error handling with detailed error messages:

- **400 Bad Request**: Invalid input or missing required fields
- **404 Not Found**: Data or resource not found
- **422 Unprocessable Entity**: Data processing or analysis errors
- **500 Internal Server Error**: Unexpected server errors
- **503 Service Unavailable**: LLM service errors

## Performance Considerations

### Data Optimization
- Automatic data type optimization reduces memory usage
- Efficient storage using pickle format
- Lazy loading of large datasets

### Memory Management
- Configurable file size limits
- Automatic cleanup of temporary files
- Memory-efficient statistical computations

### Scalability
- Stateless design for horizontal scaling
- Persistent data storage
- Configurable resource limits

## Security

### Input Validation
- File type and size validation
- SQL injection prevention
- Input sanitization

### API Security
- CORS configuration
- Request size limits
- Error message sanitization

## Development

### Project Structure
```
fastapi_statistical_bot/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── dependencies.py      # Dependency injection
│   └── api/
│       ├── endpoints/       # API endpoints
│       └── models/          # Pydantic models
├── core/                    # Business logic
│   ├── data_handler.py      # Data processing
│   ├── statistical_analyzer.py # Statistical analysis
│   ├── llm_agent.py         # LLM integration
│   └── storage.py           # Data persistence
├── utils/                   # Utilities
│   ├── exceptions.py        # Custom exceptions
│   └── logging.py           # Logging configuration
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

### Adding New Features

1. **New Statistical Method**: Add to `StatisticalAnalyzer` class
2. **New Endpoint**: Create in appropriate endpoint module
3. **New Data Type**: Extend `DataHandler` class
4. **New LLM Feature**: Modify `StatisticalLLMAgent` class

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

## Deployment

### Production Deployment

1. **Set production environment variables**
```bash
DEBUG=false
LOG_LEVEL=WARNING
CORS_ORIGINS=["https://yourdomain.com"]
```

2. **Use production ASGI server**
```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

3. **Set up reverse proxy (nginx)**
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "app.main"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the API documentation at `/docs`
- Review the error messages for troubleshooting
- Ensure OpenAI API key is properly configured
- Verify data format compatibility

## Changelog

### v1.0.0
- Initial release
- Complete statistical analysis suite
- LLM integration with OpenAI
- Production-ready FastAPI implementation
- Comprehensive documentation and error handling

