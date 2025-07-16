#!/usr/bin/env python
"""LLM agent for natural language query processing with PandasAI integration."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Generator, Optional
from openai import OpenAI
from pandasql import sqldf

from app.config import settings
from core.statistical_analyzer import StatisticalAnalyzer
from utils.exceptions import LLMError, ValidationError
from utils.logging import LoggerMixin

from pandasai import SmartDatalake
from pandasai.llm.base import LLM


class OpenAILLM(LLM):
    """Custom OpenAI LLM wrapper that handles both str and object prompts."""

    def __init__(self, api_token: str, model: str = "gpt-4o", **kwargs):
        self.client = OpenAI(api_key=api_token)
        self.model = model
        super().__init__(**kwargs)

    def call(self, instruction, value: str = "", suffix: str = "") -> str:
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
    """LLM-powered agent for processing natural language and SQL queries across DataFrames."""

    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        self.dataframes = dataframes
        self.analyzers = {name: StatisticalAnalyzer(df) for name, df in dataframes.items()}
        self.llm = OpenAILLM(api_token=settings.openai_api_key)

        self.smart_lake = SmartDatalake(
            [df for df in dataframes.values()],
            config={
                "llm": self.llm,
                "enable_cache": True,
                "use_error_correction_framework": True,
                "conversational": False,
                "save_charts": True,
                "verbose": True
            }
        )

        self.conversation_history: List[Dict[str, str]] = []
        self.log_operation("initialized", datasets=list(self.dataframes.keys()))

    def process_query(
        self,
        query: str,
        df_name: Optional[str] = None,
        sql_mode: bool = False
    ) -> Generator[str, None, None]:
        self.log_operation("process_query", query=query, dataset=df_name or "[all]", sql_mode=sql_mode)
        self.conversation_history.append({"role": "user", "content": query})
        try:
            if sql_mode:
                df = sqldf(query, self.dataframes)
                if df.empty:
                    yield "ℹ️ No rows returned by SQL query."
                else:
                    for row in df.head(10).to_dict(orient="records"):
                        yield str(row)
                response = f"SQL returned {len(df)} rows."

            else:
                prompt = f"Using dataset '{df_name}': {query}" if df_name else query
                response = self.smart_lake.chat(prompt)
                yield response

            self.conversation_history.append({"role": "assistant", "content": response})

        except Exception as e:
            err = f"❗️ Error: {e}"
            self.logger.error(err)
            yield err

    def merge_dataframes(self, df1: str, df2: str, on: List[str], how: str = 'inner') -> pd.DataFrame:
        if df1 not in self.dataframes or df2 not in self.dataframes:
            raise ValidationError(f"DataFrame '{df1}' or '{df2}' not found.")
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
            "Join two tables on a key using SQL mode."
        ]
        if analyzer.numeric_columns:
            col = analyzer.numeric_columns[0]
            suggestions += [
                f"What is the standard deviation of {col}?",
                f"Plot histogram of {col}."
            ]
        return suggestions[:8]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()

    def clear_conversation_history(self):
        self.conversation_history.clear()
        self.log_operation("clear_conversation_history")
