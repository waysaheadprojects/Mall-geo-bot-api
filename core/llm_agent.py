import os
import json
import logging
from typing import Generator, Dict, Any, List

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasAIOpenAI
from openai import OpenAI
from core.statistical_analyzer import StatisticalAnalyzer

class LLMError(Exception):
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatisticalLLMAgent:
    def __init__(self, df: pd.DataFrame, analyzer: StatisticalAnalyzer):
        self.df = df
        self.analyzer = analyzer
        self.conversation_history: List[Dict[str, str]] = []

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY not found in environment.")

        self.client = OpenAI(api_key=api_key)
        self.pandasai_llm = PandasAIOpenAI(api_token=api_key)

        self.smart_dataframe = SmartDataframe(df, config={
            "llm": self.pandasai_llm,
            "enable_cache": False,
            "save_charts": True,
            "verbose": True,
            "show_code": True,
            "enable_retries": True,
        })

        self.dataset_context = self._generate_dataset_context()

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

    def _get_intent(self, query: str) -> Dict[str, Any]:
        prompt = f"""
Determine if the following query is clear or ambiguous.

If clear, respond with:
{{"type": "clear_query"}}

If ambiguous, respond with:
{{"type": "clarification", "question": "Your clarifying question"}}

Query: "{query}"
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            return {"type": "clear_query"}

    def process_query(self, query: str) -> Generator[str, None, None]:
        logger.info(f"Processing query: {query}")
        try:
            intent = self._get_intent(query)

            if intent.get("type") == "clarification":
                clarification = intent.get("question", "Could you clarify your query?")
                self.conversation_history.append({"role": "assistant", "content": clarification})
                yield clarification
                return

            self.conversation_history.append({"role": "user", "content": query})
            result = self.smart_dataframe.chat(query)

            # Parse results
            if isinstance(result, pd.DataFrame):
                result_str = result.to_markdown(index=False)
            elif isinstance(result, str):
                if result.strip().endswith(".png"):
                    chart_url = f"https://mallgpt.waysaheadglobal.com/{result.replace(os.sep, '/')}"
                    result_str = f"[CHART] {chart_url}"
                else:
                    result_str = result.strip()
            else:
                result_str = str(result)

            explanation_prompt = f"""
You are a business analyst assistant.
Your job is to explain data analysis results in HTML for a business audience.

Instructions:
- Summarize the result clearly in 2-3 sentences.
- If a chart URL is mentioned as [CHART] https://..., embed it with <img src>.
- If markdown tables are present, convert them to HTML tables.
- Do not include any Python code or technical explanation.

Result:
{result_str}

Output the explanation as clean HTML.
"""

            gpt_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": explanation_prompt}],
                temperature=0.4,
                max_tokens=1000
            )

            final_output = gpt_response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": final_output})
            yield final_output

        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            yield "⚠️ Sorry, I encountered an error while processing your query."

    def suggest_questions(self) -> List[str]:
        suggestions = [
            "Which tenants have the largest deal sizes?",
            "Show a distribution of rental rates by floor.",
            "How many units are vacant versus leased?",
            "What’s the average chargeable area per unit?",
            "Which brands have the highest total area leased?"
        ]

        if self.analyzer and self.analyzer.numeric_columns:
            col1 = self.analyzer.numeric_columns[0]
            suggestions.extend([
                f"What is the mean and standard deviation of {col1}?",
                f"Are there any significant outliers in {col1}?"
            ])
            if len(self.analyzer.numeric_columns) > 1:
                col2 = self.analyzer.numeric_columns[1]
                suggestions.append(f"Is there a correlation between {col1} and {col2}?")

        return suggestions[:8]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history

    def clear_conversation_history(self):
        self.conversation_history.clear()
        logger.info("Cleared conversation history.")
