"""LLM agent for advanced statistical and relational query processing with PandasAI and pandasql integration."""

import pandas as pd
from typing import Dict, Any, List, Generator, Optional, Union
from openai import OpenAI
from app.config import settings
from core.statistical_analyzer import StatisticalAnalyzer
from utils.exceptions import LLMError
from utils.logging import LoggerMixin
from pandasai import SmartDataframe
from pandasai.llm.base import LLM
from pandasql import sqldf


class OpenAILLM(LLM):
    """Custom OpenAI LLM wrapper for PandasAI."""

    def __init__(self, api_token: str, model: str = "gpt-4o", **kwargs):
        self.api_token = api_token
        self.model = model
        self.client = OpenAI(api_key=api_token)
        super().__init__(**kwargs)

    def call(self, instruction: str, value: str = "", suffix: str = "") -> str:
        prompt = f"{instruction}\n{value}\n{suffix}".strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI API call failed: {str(e)}")

    @property
    def type(self) -> str:
        return "openai"


class StatisticalLLMAgent(LoggerMixin):
    """Advanced LLM-powered agent for statistical and relational data analysis using multiple DataFrames."""

    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        self.dataframes = dataframes
        self.analyzers = {name: StatisticalAnalyzer(df) for name, df in dataframes.items()}
        self.client = self._initialize_openai_client()
        self.conversation_history = []

        self.pandas_ai_llm = OpenAILLM(api_token=settings.openai_api_key)
        self.smart_dataframes = {
            name: SmartDataframe(
                df,
                config={
                    "llm": self.pandas_ai_llm,
                    "enable_cache": True,
                    "use_error_correction_framework": True,
                    "conversational": False,
                    "save_charts": True,
                    "verbose": True,
                    "custom_head": self._generate_dataset_context(name, self.analyzers[name])
                },
            )
            for name, df in dataframes.items()
        }

        for name, analyzer in self.analyzers.items():
            self.log_operation(
                "initialized",
                dataset=name,
                rows=len(analyzer.df),
                columns=len(analyzer.df.columns),
                numeric_cols=len(analyzer.numeric_columns),
                categorical_cols=len(analyzer.categorical_columns),
            )

    def _initialize_openai_client(self) -> OpenAI:
        if not settings.openai_api_key:
            raise LLMError("OpenAI API key not configured")
        return OpenAI(api_key=settings.openai_api_key)

    def _generate_dataset_context(self, name: str, analyzer: StatisticalAnalyzer) -> str:
        try:
            basic_info = analyzer.get_basic_info()
            schema_overview = analyzer.get_schema_overview()
        except Exception:
            return ""

        context = f"Dataset Summary:\n- Rows: {basic_info['total_rows']:,}\n- Columns: {basic_info['total_columns']}\n- Numeric Columns: {', '.join(analyzer.numeric_columns[:5])}\n- Categorical Columns: {', '.join(analyzer.categorical_columns[:5])}\n"
        for col_info in schema_overview["columns"][:10]:
            context += f"- {col_info['column']} ({col_info['dtype']}): {col_info['null_percentage']:.1f}% null, {col_info['unique_count']} unique\n"
        return context.strip()

    def process_query(self, query: str, df_name: Optional[str] = None, sql_mode: bool = False) -> Generator[str, None, None]:
        self.log_operation("process_query", query_length=len(query), dataset=df_name or "[all]", sql_mode=sql_mode)

        try:
            self.conversation_history.append({"role": "user", "content": query})

            if sql_mode:
                result = self._execute_sql(query)
            elif df_name:
                sdf = self.smart_dataframes.get(df_name)
                if sdf is None:
                    raise ValueError(f"No DataFrame named '{df_name}' found.")
                result = sdf.chat(query)
            else:
                result = self._process_across_dataframes(query)

            if isinstance(result, pd.DataFrame):
                response_content = result.to_html(index=False) if not result.empty else "No matching data found."
            elif isinstance(result, (int, float, str, bool)):
                response_content = f"Answer: {result}"
            else:
                response_content = "Unrecognized result format."

            for chunk in self._stream_text(response_content):
                yield chunk

            self.conversation_history.append({"role": "assistant", "content": response_content})
            self.conversation_history = self.conversation_history[-20:]

        except Exception as e:
            error_message = f"Processing failed: {str(e)}"
            self.logger.error(error_message)
            yield error_message

    def _execute_sql(self, query: str) -> pd.DataFrame:
        local_ns = {name: df for name, df in self.dataframes.items()}
        return sqldf(query, local_ns)

    def _process_across_dataframes(self, query: str) -> Union[str, pd.DataFrame]:
        context = "\n\n".join([
            f"Dataset: {name}\n{self._generate_dataset_context(analyzer)}"
            for name, analyzer in self.analyzers.items()
        ])
        instruction = f"You have multiple datasets. {context}\n\nBased on this context, answer: {query}"
        return self.smart_dataframes[list(self.smart_dataframes.keys())[0]].llm.call(instruction)

    def _stream_text(self, text: str, chunk_size: int = 60) -> Generator[str, None, None]:
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()

    def clear_conversation_history(self) -> None:
        self.conversation_history = []
        self.log_operation("clear_conversation_history")

    def add_dataframe(self, name: str, df: pd.DataFrame):
        self.dataframes[name] = df
        self.analyzers[name] = StatisticalAnalyzer(df)
        self.smart_dataframes[name] = SmartDataframe(df, config={"llm": self.pandas_ai_llm, "enable_cache": False})
        self.log_operation("add_dataframe", name=name, rows=len(df), columns=len(df.columns))

    def remove_dataframe(self, name: str):
        if name in self.dataframes:
            del self.dataframes[name]
            del self.analyzers[name]
            del self.smart_dataframes[name]
            self.log_operation("remove_dataframe", name=name)

    def merge_dataframes(self, df1_name: str, df2_name: str, on_columns: List[str], how: str = 'inner') -> Optional[pd.DataFrame]:
        if df1_name not in self.dataframes or df2_name not in self.dataframes:
            raise ValueError(f"Both dataframes '{df1_name}' and '{df2_name}' must exist to perform a merge.")

        df1 = self.dataframes[df1_name]
        df2 = self.dataframes[df2_name]

        try:
            merged_df = pd.merge(df1, df2, on=on_columns, how=how)
            self.log_operation("merge_dataframes", df1=df1_name, df2=df2_name, on=on_columns, how=how, merged_rows=len(merged_df))
            return merged_df
        except Exception as e:
            raise RuntimeError(f"Error merging dataframes: {e}")

    def suggest_questions(self, df_name: Optional[str] = None) -> List[str]:
        if df_name and df_name in self.analyzers:
            analyzer = self.analyzers[df_name]
        else:
            analyzer = next(iter(self.analyzers.values()))

        suggestions = [
            "What are the most common categories in the data?",
            "Calculate the average value for a numeric column.",
            "Join this dataset with another using a common key.",
            "Show top 10 rows with highest values in a column.",
            "Run SQL: SELECT * FROM dataset_name WHERE column > 100"
        ]

        if analyzer.numeric_columns:
            col = analyzer.numeric_columns[0]
            suggestions += [
                f"What is the standard deviation of {col}?",
                f"Plot histogram of {col}."
            ]

        return suggestions[:8]
