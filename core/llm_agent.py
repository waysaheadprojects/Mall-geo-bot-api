"""LLM agent for natural language query processing with PandasAI integration."""

import pandas as pd
from typing import Dict, Any, List, Generator, Optional
import openai
from openai import OpenAI
import os

from app.config import settings
from core.statistical_analyzer import StatisticalAnalyzer
from utils.exceptions import LLMError, ValidationError
from utils.logging import LoggerMixin

from pandasai import SmartDataframe
from pandasai.llm.base import LLM


class OpenAILLM(LLM):
    """Custom OpenAI LLM wrapper for PandasAI."""

    def __init__(self, api_token: str, **kwargs):
        self.api_token = api_token
        self.client = OpenAI(api_key=api_token)
        super().__init__(**kwargs)

    def call(self, instruction: str, value: str = "", suffix: str = "") -> str:
        """Call the OpenAI API."""
        prompt = f"{instruction}\n{value}\n{suffix}".strip()

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI API call failed: {str(e)}")

    @property
    def type(self) -> str:
        return "openai"


class StatisticalLLMAgent(LoggerMixin):
    """LLM-powered agent for processing natural language queries about statistical data."""

    def __init__(self, df: pd.DataFrame, analyzer: StatisticalAnalyzer):
        self.df = df
        self.analyzer = analyzer
        self.client = self._initialize_openai_client()
        self.conversation_history = []

        self.pandas_ai_llm = OpenAILLM(api_token=settings.openai_api_key)
        self.smart_dataframe = SmartDataframe(df, config={"llm": self.pandas_ai_llm, "enable_cache": False, "save_charts": True})

        self.dataset_context = self._generate_dataset_context()

        self.log_operation(
            "initialized",
            rows=len(df),
            columns=len(df.columns),
            numeric_cols=len(analyzer.numeric_columns),
            categorical_cols=len(analyzer.categorical_columns)
        )

    def _initialize_openai_client(self) -> OpenAI:
        if not settings.openai_api_key:
            raise LLMError("OpenAI API key not configured")
        return OpenAI(api_key=settings.openai_api_key)

    def _generate_dataset_context(self) -> str:
        basic_info = self.analyzer.get_basic_info()
        schema_overview = self.analyzer.get_schema_overview()

        context = f"""
Dataset Overview:
- Total rows: {basic_info['total_rows']:,}
- Total columns: {basic_info['total_columns']}
- Numeric columns: {len(self.analyzer.numeric_columns)} ({', '.join(self.analyzer.numeric_columns[:10])})
- Categorical columns: {len(self.analyzer.categorical_columns)} ({', '.join(self.analyzer.categorical_columns[:10])})
- Missing values: {basic_info['missing_values']:,}

Column Details:
"""
        for col_info in schema_overview["columns"][:15]:
            context += f"- {col_info['column']}: {col_info['dtype']}, {col_info['null_percentage']:.1f}% null, {col_info['unique_count']} unique values\n"

        if len(schema_overview["columns"]) > 15:
            context += f"... and {len(schema_overview['columns']) - 15} more columns\n"

        return context

    def process_query(self, query: str) -> Generator[str, None, None]:
        self.log_operation("process_query", query_length=len(query))

        try:
            self.conversation_history.append({"role": "user", "content": query})

            pandasai_response = self.smart_dataframe.chat(query)

            if isinstance(pandasai_response, pd.DataFrame):
                response_content = f"Here's the result:<br><br>{pandasai_response.to_html(index=False)}"
            elif isinstance(pandasai_response, (int, float, bool)):
                response_content = f"The answer is: <b>{pandasai_response}</b>"
            elif isinstance(pandasai_response, str):
                if pandasai_response.strip().endswith(".png") and os.path.exists(pandasai_response):
                    response_content = f'<img src="{pandasai_response}" alt="Generated Chart" style="max-width:100%;">'
                else:
                    response_content = pandasai_response
            elif pandasai_response is None:
                response_content = "I processed your request, but couldn't generate a specific answer. Please try rephrasing your question."
            else:
                response_content = f"Result: {str(pandasai_response)}"

            for chunk in self._stream_text(response_content):
                yield chunk

            self.conversation_history.append({"role": "assistant", "content": response_content})
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

        except Exception as e:
            error_message = f"I encountered an error while analyzing your data with PandasAI: {str(e)}"
            self.logger.error(f"Query processing failed with PandasAI: {str(e)}")
            yield error_message

    def _stream_text(self, text: str, chunk_size: int = 50) -> Generator[str, None, None]:
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()

    def clear_conversation_history(self) -> None:
        self.conversation_history = []
        self.log_operation("clear_conversation_history")

    def suggest_questions(self) -> List[str]:
        suggestions = [
            "Which brands or tenants have the largest retail footprint (deal size in sft) in Inorbit Mall Hyderabad?",
            "What is the distribution of deal sizes (in square feet) for retail spaces in Inorbit Mall Hyderabad?",
            "How many retail units are currently occupied versus vacant in Inorbit Mall Hyderabad?",
            "What is the average number of seats available in retail units within Inorbit Mall Hyderabad?",
            "What is the breakdown of deal types (e.g., new lease, renewal) within Inorbit Mall Hyderabad?"
        ]

        if self.analyzer.numeric_columns:
            col = self.analyzer.numeric_columns[0]
            suggestions.extend([
                f"What is the mean and standard deviation of {col}?",
                f"Are there outliers in {col}?",
                f"What is the distribution of {col}?"
            ])

        if len(self.analyzer.numeric_columns) >= 2:
            col1, col2 = self.analyzer.numeric_columns[:2]
            suggestions.append(f"Is there a correlation between {col1} and {col2}?")

        return suggestions[:8]
