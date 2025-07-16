"""Main FastAPI application for the Statistical Analysis Bot."""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import time
import traceback

from app.config import settings
from app.api.endpoints import data, analysis, chat
from app.api.models.responses import HealthResponse, ErrorResponse, StatusEnum
from utils.logging import setup_logging, get_logger
from utils.exceptions import StatisticalBotException

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A production-ready API for statistical analysis with LLM-powered natural language queries",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# Global exception handler for custom exceptions
@app.exception_handler(StatisticalBotException)
async def statistical_bot_exception_handler(request: Request, exc: StatisticalBotException):
    """Handle custom Statistical Bot exceptions."""
    logger.error(f"Statistical Bot Exception: {exc.message}", extra={"details": exc.details})
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.message,
            error_code=exc.__class__.__name__,
            details=exc.details
        ).dict()
    )

# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", extra={"traceback": traceback.format_exc()})
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            message="An unexpected error occurred",
            error_code="InternalServerError",
            details={"error": str(exc)} if settings.debug else {}
        ).dict()
    )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Include routers
app.include_router(data.router)
app.include_router(analysis.router)
app.include_router(chat.router)

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        uptime="Service is running"
    )

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Statistical Analysis Bot API",
        "docs_url": "/docs" if settings.debug else "Documentation disabled in production",
        "health_url": "/health",
        "endpoints": {
            "data_management": "/api/data",
            "statistical_analysis": "/api/analysis", 
            "chat_llm": "/api/chat"
        }
    }

# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        ## Statistical Analysis Bot API
        
        A production-ready FastAPI application for statistical analysis with LLM-powered natural language queries.
        
        ### Features
        - **Data Upload & Processing**: Upload CSV/Excel files with automatic data type optimization
        - **Statistical Analysis**: Comprehensive statistical analysis including descriptive stats, correlation, outlier detection, distribution analysis, and hypothesis testing
        - **LLM Integration**: Natural language queries powered by OpenAI GPT models
        - **Data Persistence**: Automatic data storage and retrieval
        - **Production Ready**: Comprehensive error handling, logging, and validation
        
        ### Getting Started
        1. Upload your data using `/api/data/upload`
        2. Explore your data with `/api/data/info` and `/api/data/columns`
        3. Perform statistical analysis using `/api/analysis/*` endpoints
        4. Ask natural language questions using `/api/chat/query`
        
        ### Authentication
        Requires OpenAI API key configuration for LLM features.
        """,
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "Health",
            "description": "Health check and service status"
        },
        {
            "name": "Data Management", 
            "description": "Upload, process, and manage datasets"
        },
        {
            "name": "Statistical Analysis",
            "description": "Comprehensive statistical analysis tools"
        },
        {
            "name": "Chat & LLM",
            "description": "Natural language queries and AI-powered insights"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"CORS origins: {settings.cors_origins}")
    
    # Verify OpenAI configuration
    if not settings.openai_api_key:
        logger.warning("OpenAI API key not configured - LLM features will be unavailable")
    else:
        logger.info("OpenAI API key configured - LLM features available")
    
    # Ensure data storage directory exists
    settings.data_storage_dir.mkdir(exist_ok=True)
    logger.info(f"Data storage directory: {settings.data_storage_dir}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info(f"Shutting down {settings.app_name}")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

