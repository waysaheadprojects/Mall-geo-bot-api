#!/usr/bin/env python
"""LLM agent with PandasAI + SmartDatalake + pandasql, fixed for code-generation prompts."""

import pandas as pd
import numpy as np
from typing import Dict, List, Generator, Optional, Union

from openai import OpenAI
from pandasql import sqldf

from app.config import settings
from core.statistical_analyzer import StatisticalAnalyzer
from utils.exceptions import LLMError
from utils.logging import LoggerMixin

from pandasai import SmartDatalake
from pandasai.llm.base import LLM


class OpenAILLM(LLM):
    """OpenAI LLM wrapper that supports both string and prompt-like instructions."""

    def __init__(self, api_token: str, model: str = "gpt-4o", **kwargs):
        self.client = OpenAI(api_key=api_token)
        self.model = model
        super().__init__(**kwargs)

    def call(self, instruction, value: str = "", suffix: str = "") -> str:
        # Handle non-string prompts (e.g. GeneratePythonCodePrompt)
        if not isinstance(instruction, str):
            instruction_text = getattr(instruction, "prompt", str(instruction))
        else:
            instruction_text = instruction

        prompt = "\n".join(filter(None, [instruction_text, value, suffix]))

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI API call failed: {e}")

    @property
    def type(self) -> str:
        return "openai"


class StatisticalLLMAgent(LoggerMixin):
    """
    Agent for statistical and SQL queries across multiple DataFrames, with streaming and full pipeline support.
    """

    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        self.dataframes = dataframes
        self.analyzers = {name: StatisticalAnalyzer(df) for name, df in dataframes.items()}
        self.llm = OpenAILLM(api_token=settings.openai_api_key)

        # SmartDatalake for multi-DataFrame conversational querying
        self.smart_lake = SmartDatalake(list(dataframes.values()), config={
            "llm": self.llm,
            "enable_cache": True,
            "use_error_correction_framework": True,
            "conversational": False,
            "save_charts": True,
            "verbose": True
        })

        self.history: List[Dict[str, str]] = []
        self.log_operation("initialized", datasets=list(self.dataframes.keys()))

    def process_query(
        self,
        query: str,
        df_name: Optional[str] = None,
        sql_mode: bool = False
    ) -> Generator[str, None, None]:
        self.log_operation("process_query", query=query, df=df_name or "[all]", sql_mode=sql_mode)
        self.history.append({"role": "user", "content": query})

        try:
            if sql_mode:
                df = self._run_sql(query)
                yield from self._stream_df(df)
                response = f"SQL returned {len(df)} rows."
            else:
                response = self._chat_query(query, df_name)
                yield response

            self.history.append({"role": "assistant", "content": response})

        except Exception as e:
            err = f"❗️ Error: {e}"
            self.logger.error(err)
            yield err

    def _run_sql(self, query: str) -> pd.DataFrame:
        try:
            return sqldf(query, self.dataframes)
        except Exception as e:
            raise LLMError(f"SQL execution failed: {e}")

    def _chat_query(self, query: str, df_name: Optional[str]) -> str:
        if df_name:
            if df_name not in self.dataframes:
                raise ValueError(f"No DataFrame named '{df_name}'.")
            return self.smart_lake.chat(f"Using '{df_name}': {query}")
        return self.smart_lake.chat(query)

    def _stream_df(self, df: pd.DataFrame) -> Generator[str, None, None]:
        if df.empty:
            yield "ℹ️ SQL query returned no results."
        else:
            for row in df.head(10).to_dict(orient="records"):
                yield str(row)

    def merge_dataframes(self, df1: str, df2: str, on: List[str], how: str = 'inner') -> pd.DataFrame:
        if df1 not in self.dataframes or df2 not in self.dataframes:
            raise ValueError(f"One or both dataframes '{df1}', '{df2}' not found.")
        merged = pd.merge(self.dataframes[df1], self.dataframes[df2], on=on, how=how)
        self.log_operation("merge", left=df1, right=df2, on=on, how=how, rows=len(merged))
        return merged

    def list_dataframes(self) -> List[str]:
        return list(self.dataframes.keys())

    def suggest_questions(self, df_name: Optional[str] = None) -> List[str]:
        analyzer = self.analyzers.get(df_name, next(iter(self.analyzers.values())))
        suggestions = [
            "Show top 5 rows of the dataset.",
            "What is the mean and std of a numeric column?",
            "Plot histogram of a numeric column.",
            "Join two tables on a key, using SQL mode."
        ]
        if analyzer.numeric_columns:
            col = analyzer.numeric_columns[0]
            suggestions += [
                f"What is the standard deviation of {col}?",
                f"Plot histogram of {col}."
            ]
        return suggestions[:8]

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def clear_history(self):
        self.history.clear()
        self.log_operation("clear_history")
