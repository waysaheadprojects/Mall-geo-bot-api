"""Chat and LLM query endpoints for the Statistical Analysis Bot API."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import json
import asyncio

from app.dependencies import get_llm_agent, get_current_data
from app.api.models.requests import ChatQueryRequest
from app.api.models.responses import (
    ChatQueryResponse, ChatHistoryResponse, SuggestionsResponse,
    StatusEnum, ChatMessage, BaseResponse
)
from core.llm_agent import StatisticalLLMAgent
from core.storage import storage
from utils.exceptions import LLMError
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat & LLM"])


@router.post("/query", response_model=ChatQueryResponse)
async def process_chat_query(
    request: ChatQueryRequest,
    llm_agent: StatisticalLLMAgent = Depends(get_llm_agent)
):
    """
    Process a natural language query about the data using LLM.
    
    Returns a comprehensive response with analysis and insights.
    """
    try:
        logger.info(f"Processing chat query: {request.query[:100]}...")
        
        # Process the query and collect the full response
        full_response = ""
        analysis_performed = []
        
        try:
            for chunk in llm_agent.process_query(request.query):
                full_response += chunk
            
            # Determine what analyses were performed based on the query
            query_lower = request.query.lower()
            if any(word in query_lower for word in ["mean", "average", "median", "std", "statistics"]):
                analysis_performed.append("descriptive_statistics")
            if any(word in query_lower for word in ["correlation", "relationship"]):
                analysis_performed.append("correlation_analysis")
            if any(word in query_lower for word in ["outlier", "anomaly"]):
                analysis_performed.append("outlier_detection")
            if any(word in query_lower for word in ["distribution", "normal"]):
                analysis_performed.append("distribution_analysis")
            if any(word in query_lower for word in ["test", "hypothesis"]):
                analysis_performed.append("hypothesis_testing")
            
            if not analysis_performed:
                analysis_performed.append("general_inquiry")
                
        except Exception as e:
            logger.error(f"LLM processing error: {str(e)}")
            full_response = f"I encountered an error while processing your query: {str(e)}"
            analysis_performed = ["error"]
        
        # Get conversation history if requested
        conversation_history = None
        if request.include_history:
            history = llm_agent.get_conversation_history()
            conversation_history = [
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in history
            ]
        
        # Get suggested questions
        suggested_questions = llm_agent.suggest_questions()
        
        # Save chat history to storage
        try:
            current_history = llm_agent.get_conversation_history()
            storage.save_chat_history(current_history)
        except Exception as e:
            logger.warning(f"Failed to save chat history: {str(e)}")
        
        return ChatQueryResponse(
            status=StatusEnum.SUCCESS,
            message="Query processed successfully",
            response=full_response,
            conversation_history=conversation_history,
            suggested_questions=suggested_questions,
            analysis_performed=analysis_performed
        )
        
    except LLMError as e:
        logger.error(f"LLM error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process query")


@router.post("/query/stream")
async def stream_chat_query(
    request: ChatQueryRequest,
    llm_agent: StatisticalLLMAgent = Depends(get_llm_agent)
):
    """
    Process a natural language query with streaming response.
    
    Returns a streaming response for real-time interaction.
    """
    async def generate_response():
        try:
            logger.info(f"Streaming chat query: {request.query[:100]}...")
            
            # Process query and yield chunks
            for chunk in llm_agent.process_query(request.query):
                # Format as Server-Sent Events
                yield f"data: {json.dumps({'chunk': chunk, 'type': 'content'})}\n\n"
                await asyncio.sleep(0.01)  # Small delay for better streaming effect
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
            # Save chat history
            try:
                current_history = llm_agent.get_conversation_history()
                storage.save_chat_history(current_history)
            except Exception as e:
                logger.warning(f"Failed to save chat history: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            error_msg = f"Error processing query: {str(e)}"
            yield f"data: {json.dumps({'chunk': error_msg, 'type': 'error'})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(llm_agent: StatisticalLLMAgent = Depends(get_llm_agent)):
    """
    Get the complete conversation history.
    """
    try:
        history = llm_agent.get_conversation_history()
        
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in history
        ]
        
        return ChatHistoryResponse(
            status=StatusEnum.SUCCESS,
            message="Chat history retrieved successfully",
            history=chat_messages,
            message_count=len(chat_messages)
        )
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve chat history")


@router.delete("/history", response_model=BaseResponse)
async def clear_chat_history(llm_agent: StatisticalLLMAgent = Depends(get_llm_agent)):
    """
    Clear the conversation history.
    """
    try:
        llm_agent.clear_conversation_history()
        storage.clear_chat_history()
        
        return BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Chat history cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear chat history")


@router.get("/suggestions", response_model=SuggestionsResponse)
async def get_query_suggestions(llm_agent: StatisticalLLMAgent = Depends(get_llm_agent)):
    """
    Get suggested questions based on the current dataset.
    """
    try:
        suggestions = llm_agent.suggest_questions()
        
        # Categorize suggestions
        categories = {
            "descriptive": [
                "What are the basic statistics for the numeric columns?",
                "Can you provide a summary of the dataset?"
            ],
            "correlation": [
                "Are there any strong correlations in the data?",
                "What relationships exist between variables?"
            ],
            "outliers": [
                "Can you identify any outliers in the dataset?",
                "Are there any unusual values in the data?"
            ],
            "distribution": [
                "What does the distribution of the data look like?",
                "Are the variables normally distributed?"
            ],
            "general": [
                "What insights can you provide about this data?",
                "What patterns do you see in the data?"
            ]
        }
        
        # Add column-specific suggestions if available
        if hasattr(llm_agent.analyzer, 'numeric_columns') and llm_agent.analyzer.numeric_columns:
            col = llm_agent.analyzer.numeric_columns[0]
            categories["column_specific"] = [
                f"What is the distribution of {col}?",
                f"Are there outliers in {col}?",
                f"What are the statistics for {col}?"
            ]
        
        return SuggestionsResponse(
            status=StatusEnum.SUCCESS,
            message="Query suggestions generated successfully",
            suggestions=suggestions,
            categories=categories
        )
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate suggestions")


@router.get("/context")
async def get_chat_context(data: tuple = Depends(get_current_data)):
    """
    Get context information about the current dataset for chat interface.
    """
    try:
        df, metadata = data
        
        context = {
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "file_names": metadata.get("file_names", []),
                "upload_time": metadata.get("upload_time")
            },
            "column_info": {
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "total_columns": len(df.columns)
            },
            "data_quality": {
                "missing_values": int(df.isnull().sum().sum()),
                "missing_percentage": float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                "duplicate_rows": int(df.duplicated().sum())
            }
        }
        
        return {
            "status": "success",
            "message": "Chat context retrieved successfully",
            "context": context
        }
        
    except Exception as e:
        logger.error(f"Error getting chat context: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve chat context")

