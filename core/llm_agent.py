import os
import json
import logging
from typing import Generator, Dict, Any, List

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAIOpenAI
from openai import OpenAI
from core.statistical_analyzer import StatisticalAnalyzer

# Custom exception
class LLMError(Exception):
    pass

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom OpenAI Wrapper
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


# Final StatisticalLLMAgent
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
            "save_charts": True
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
You are an expert data analyst. Your job is to answer questions about the provided dataset.
- If the query is ambiguous, ask a clarifying question.
- If it is clear, answer using relevant statistics or code.
- Use charts if helpful and describe them clearly.
"""
        examples = """
Examples:
User: "What's the average deal size?"
Assistant: "The average deal size is 1,200 sqft."

User: "Show me the biggest tenants."
Assistant: "Do you mean by total area or by number of units?"
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
            logger.warning("Intent check failed (invalid JSON). Proceeding as clear.")
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

            pandasai_response = self.smart_dataframe.chat(full_prompt)

            if isinstance(pandasai_response, pd.DataFrame):
                html = pandasai_response.to_html(index=False)
                self.conversation_history.append({"role": "assistant", "content": html})
                yield html
            elif isinstance(pandasai_response, str) and pandasai_response.strip().endswith(".png"):
                url = f"https://mallgpt.waysaheadglobal.com/{pandasai_response.replace(os.sep, '/')}"
                img_html = f'<img src="{url}" alt="Chart" style="max-width:100%;">'
                self.conversation_history.append({"role": "assistant", "content": img_html})
                yield img_html
            elif isinstance(pandasai_response, str):
                self.conversation_history.append({"role": "assistant", "content": pandasai_response})
                yield pandasai_response
            else:
                fallback = str(pandasai_response)
                self.conversation_history.append({"role": "assistant", "content": fallback})
                yield fallback

        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            yield "⚠️ Sorry, I encountered an error while processing your query."

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history

    def clear_conversation_history(self):
        self.conversation_history.clear()
        logger.info("Cleared conversation history.")
