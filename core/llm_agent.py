import os
import json
import logging
from typing import Generator, Dict, Any, List

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAIOpenAI
from openai import OpenAI
from core.statistical_analyzer import StatisticalAnalyzer

class LLMError(Exception):
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAILLM:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY not found in environment.")
        self.client = OpenAI(api_key=api_key)

    def call(self, instruction: str, value: str = "", suffix: str = "") -> str:
        prompt = f"{instruction}\n{value}\n{suffix}".strip()
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI API call failed: {str(e)}")

    def stream_call(self, prompt: str) -> Generator[str, None, None]:
        try:
            stream = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"LLM streaming failed: {str(e)}")


class StatisticalLLMAgent:
    def __init__(self, df: pd.DataFrame, analyzer: StatisticalAnalyzer):
        self.df = df
        self.analyzer = analyzer
        self.conversation_history: List[Dict[str, str]] = []

        self.llm_wrapper = OpenAILLM()
        self.client = self.llm_wrapper.client

        self.pandasai_llm = PandasAIOpenAI(api_token=os.environ.get("OPENAI_API_KEY"))
        self.smart_dataframe = SmartDataframe(df, config={
            "llm": self.pandasai_llm,
            "enable_cache": False,
            "save_charts": True,
            "verbose": True,
            "show_code": True,
            "enable_retries": True,
            "custom_head": "Explain the approach before executing code.",
            "custom_tail": "Ensure output includes reasoning, code used, and final results in readable format."
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

    def _build_prompt(self, query: str) -> str:
        system_prompt = """
You are a data analyst. Provide clear, structured explanations in your answers.
Avoid showing code unless explicitly asked.
"""
        examples = """
Examples:
User: "What's the average rent?"
Assistant: "The average rent is ₹4,125 per sq ft, based on 2,703 deals."

User: "Show deal size distribution"
Assistant: "Here’s a histogram of deal sizes. Most deals cluster between 2,000–5,000 sq ft."
"""
        history = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in self.conversation_history])
        return f"{system_prompt}\n\nDataset Context:\n{self.dataset_context}\n\nConversation:\n{history}\n\nExamples:\n{examples}\n\nUser Query: \"{query}\""

    def _get_intent(self, query: str) -> Dict[str, Any]:
        intent_prompt = """
Check if the query is ambiguous.

If clear:
{{"type": "clear_query"}}

If ambiguous:
{{"type": "clarification", "question": "Your clarifying question"}}

Query: "{query}"
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            logger.info(f"Intent response: {content}")
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Intent check failed. Proceeding as clear.")
            return {"type": "clear_query"}
        except Exception as e:
            logger.error(f"Intent check failed: {e}")
            return {"type": "clear_query"}

    def process_query(self, query: str) -> Generator[str, None, None]:
        logger.info(f"Processing query: {query}")
        try:
            intent = self._get_intent(query)

            if intent.get("type") == "clarification":
                question = intent.get("question", "Could you clarify your query?")
                self.conversation_history.append({"role": "assistant", "content": question})
                yield question
                return

            self.conversation_history.append({"role": "user", "content": query})
            full_prompt = self._build_prompt(query)

            # Step 1: Get result from PandasAI
            pandasai_response = self.smart_dataframe.chat(full_prompt)

            # Step 2: Convert to plain text for GPT
            if isinstance(pandasai_response, pd.DataFrame):
                result_str = f"Data:\n{pandasai_response.to_markdown(index=False)}"
            elif isinstance(pandasai_response, str):
                result_str = pandasai_response
            elif isinstance(pandasai_response, (int, float, bool)):
                result_str = f"Result: {pandasai_response}"
            else:
                result_str = str(pandasai_response)

            chart_html = ""
            if isinstance(pandasai_response, str) and pandasai_response.strip().endswith(".png"):
                chart_path = pandasai_response.strip().replace(os.sep, "/")
                public_url = f"https://mallgpt.waysaheadglobal.com/{chart_path}"
                chart_html = f"<img src='{public_url}' style='max-width:100%;'><br>"

            # Step 3: Send result to GPT for business explanation
            gpt_prompt = """
You are a business analyst assistant.

You will be given an analysis result generated from a data science tool (like PandasAI). Your job is to:
- Ignore any code or technical syntax
- Summarize the findings in clear, professional business terms
- Present the insight in simple HTML with explanation and tables
- If a chart is present, describe what it shows

Here is the analysis result:

Now, generate a clean HTML response for a business user. Do not show code. Just explain what the data shows and why it matters.
"""

            gpt_response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": gpt_prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            business_html = gpt_response.choices[0].message.content.strip()
            final_output = chart_html + business_html

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
