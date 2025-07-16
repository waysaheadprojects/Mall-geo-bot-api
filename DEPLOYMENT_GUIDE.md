# FastAPI Statistical Analysis Bot - Deployment Guide

## Quick Start

### 1. Environment Setup
```bash
# Navigate to project directory
cd fastapi_statistical_bot

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env file with your OpenAI API key
```

### 2. Run the Application
```bash
# Development mode
python -m app.main

# Production mode with Gunicorn
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 3. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Upload test data
curl -X POST "http://localhost:8000/api/data/upload" -F "files=@your_data.csv"

# Get data info
curl http://localhost:8000/api/data/info

# Perform analysis
curl -X POST "http://localhost:8000/api/analysis/descriptive" \
  -H "Content-Type: application/json" \
  -d '{"column": "your_column_name"}'
```

## Key Features Converted from Streamlit

### ✅ Data Management
- **File Upload**: Multi-file CSV/Excel upload with validation
- **Data Processing**: Automatic data type optimization and quality assessment
- **Data Persistence**: Automatic storage with metadata tracking
- **Column Analysis**: Detailed column information and statistics

### ✅ Statistical Analysis
- **Descriptive Statistics**: Complete statistical summaries
- **Correlation Analysis**: Pearson, Spearman, Kendall correlations
- **Outlier Detection**: IQR and Z-score methods
- **Distribution Analysis**: Normality tests and shape analysis
- **Hypothesis Testing**: Automatic test selection and interpretation
- **Categorical Analysis**: Frequency distributions and entropy

### ✅ LLM Integration
- **Natural Language Queries**: Ask questions about your data
- **Streaming Responses**: Real-time response generation
- **Context Awareness**: Maintains conversation history
- **Suggested Questions**: AI-powered query suggestions

### ✅ Production Features
- **Error Handling**: Comprehensive exception handling
- **Input Validation**: Pydantic models for all requests/responses
- **Logging**: Structured logging with configurable levels
- **CORS Support**: Cross-origin resource sharing
- **API Documentation**: Auto-generated OpenAPI/Swagger docs
- **Health Monitoring**: Service health and status endpoints

## API Endpoints

### Data Management
- `POST /api/data/upload` - Upload CSV/Excel files
- `GET /api/data/info` - Get dataset information
- `GET /api/data/columns` - List all columns
- `GET /api/data/status` - Check data loading status
- `DELETE /api/data/clear` - Clear stored data

### Statistical Analysis
- `POST /api/analysis/descriptive` - Descriptive statistics
- `POST /api/analysis/correlation` - Correlation analysis
- `POST /api/analysis/outliers` - Outlier detection
- `POST /api/analysis/distribution` - Distribution analysis
- `POST /api/analysis/hypothesis` - Hypothesis testing
- `POST /api/analysis/categorical` - Categorical analysis

### Chat & LLM
- `POST /api/chat/query` - Natural language queries
- `POST /api/chat/query/stream` - Streaming responses
- `GET /api/chat/history` - Conversation history
- `GET /api/chat/suggestions` - Query suggestions

## Configuration

### Required Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here  # Required for LLM features
```

### Optional Configuration
```bash
HOST=0.0.0.0                    # Server host
PORT=8000                       # Server port
DEBUG=false                     # Debug mode
LOG_LEVEL=INFO                  # Logging level
DATA_STORAGE_DIR=data_storage   # Data storage directory
MAX_FILE_SIZE=104857600         # Max file size (100MB)
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "app.main"]
```

### Build and Run
```bash
# Build image
docker build -t statistical-bot-api .

# Run container
docker run -p 8000:8000 --env-file .env statistical-bot-api
```

## Production Deployment

### 1. Use Production ASGI Server
```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 2. Set Production Environment
```bash
DEBUG=false
LOG_LEVEL=WARNING
CORS_ORIGINS=["https://yourdomain.com"]
```

### 3. Reverse Proxy (Nginx)
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

## Testing

The application has been tested with:
- ✅ Health check endpoint
- ✅ Data upload functionality
- ✅ Statistical analysis endpoints
- ✅ Error handling and validation
- ✅ Data persistence and retrieval

## Migration from Streamlit

### What Changed
1. **UI → API**: Web interface replaced with RESTful API endpoints
2. **Session State → Persistence**: Data now persists across requests
3. **Streamlit Components → Pydantic Models**: Type-safe request/response models
4. **Direct Function Calls → HTTP Endpoints**: All functionality accessible via HTTP
5. **Real-time Updates → Streaming Responses**: Maintains real-time feel with streaming

### What Stayed the Same
1. **Core Logic**: All statistical analysis functions preserved
2. **LLM Integration**: OpenAI integration maintained
3. **Data Processing**: Same data handling and optimization
4. **Analysis Capabilities**: All statistical methods available

## Support

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Status Check: `http://localhost:8000/api/data/status`

For issues:
1. Check logs for error details
2. Verify OpenAI API key configuration
3. Ensure data format compatibility
4. Review API documentation for correct usage

