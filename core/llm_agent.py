import os
import json
import logging
import re
import uuid
from typing import Generator, Dict, Any, List, Union

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI, APIError
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import safer_getattr, full_write_guard, guarded_iter_unpack_sequence, guarded_unpack_sequence
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter

# --- Configuration ---
matplotlib.use("Agg")
sns.set_theme(style="whitegrid")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass

class StatisticalLLMAgent:
    """
    An intelligent agent that answers questions about a DataFrame by generating and safely executing
    Python code. This version uses a more direct, streamlined architecture to ensure reliability.
    """
    def __init__(self, df: pd.DataFrame, analyzer=None):
        self._setup_openai_client()
        self.df = self._prepare_dataframe(df)
        self.dataset_context = self._generate_dataset_context()
        self._globals = self._create_safe_globals()
        self.conversation_history: List[Dict[str, str]] = []

    def _setup_openai_client(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY not found in environment.")
        self.client = OpenAI(api_key=api_key)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        sanitized_columns = {col: re.sub(r'[^0-9a-zA-Z_]+', '_', col) for col in df.columns}
        df = df.rename(columns=sanitized_columns)
        logger.info("DataFrame columns sanitized.")
        numeric_cols_to_clean = ['current_rent_chargeable_inr_per_sft', 'chargeable_area_sft']
        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"Cleaned numeric column: '{col}'")
        return df

    def _make_save_chart_function(self):
        def save_chart(plt_obj: matplotlib.pyplot) -> str:
            output_dir = "exports/charts"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{uuid.uuid4().hex}.png"
            path = os.path.join(output_dir, filename)
            plt_obj.savefig(path, bbox_inches="tight", dpi=150)
            plt_obj.close()
            logger.info(f"Chart saved to {path}")
            return path
        return save_chart

    def _create_safe_globals(self) -> Dict[str, Any]:
        restricted_globals = safe_globals.copy()
        restricted_globals.update({
            "_getattr_": safer_getattr, "_write_": full_write_guard,
            "_getitem_": default_guarded_getitem, "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence, "_unpack_sequence_": guarded_unpack_sequence,
            "pd": pd, "np": np, "plt": plt, "sns": sns,
            "df": self.df, "save_chart": self._make_save_chart_function()
        })
        return restricted_globals

    def _generate_dataset_context(self) -> str:
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        context = f"""
Dataset Overview:
- Total rows: {len(self.df):,}
- Total columns: {len(self.df.columns)}
- Numeric columns: {len(numeric_cols)} ({', '.join(numeric_cols[:5])}...)
- Categorical columns: {len(categorical_cols)} ({', '.join(categorical_cols[:5])}...)
Column Details (first 15):
"""
        for col, dtype in self.df.dtypes.head(15).items():
            null_pct = self.df[col].isnull().mean() * 100
            unique_count = self.df[col].nunique()
            context += f"- {col}: {dtype}, {null_pct:.1f}% null, {unique_count} unique values\n"
        return context

    def _call_openai_api(self, prompt: str, model: str = "gpt-4o", temperature: float = 0.0, json_mode: bool = False) -> str:
        try:
            kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except APIError as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise LLMError(f"OpenAI API call failed: {str(e)}")

    def _get_intent(self, query: str) -> Dict[str, Any]:
        """Checks if the query is clear or ambiguous, defaulting to clear."""
        default_intent = {"type": "clear_query"}
        intent_prompt = """
Analyze the user's query to determine if it is clear enough to be answered or if it is ambiguous.
Respond in JSON format.
- If clear, return: {{"type": "clear_query"}}
- If ambiguous, return: {{"type": "clarification", "question": "Your clarifying question."}}
User Query: "{query}"
"""
        try:
            response_str = self._call_openai_api(intent_prompt, temperature=0.0, json_mode=True)
            intent = json.loads(response_str)
            if 'type' in intent and intent['type'] in ['clear_query', 'clarification']:
                return intent
            return default_intent
        except (json.JSONDecodeError, LLMError, Exception) as e:
            logger.warning(f"Intent check failed: {e}. Defaulting to clear query.")
            return default_intent

    def _generate_and_execute_code(self, query: str) -> Union[Dict[str, Any], None]:
        """Generates and safely executes Python code directly from the user query."""
        logger.info("Step 2: Generating Python code...")
        prompt = """
You are an expert Python code generator for data analysis. Your task is to write Python code to answer the user's question.

**DataFrame Information:**
{self.dataset_context}

**User Question:** "{query}"

**CRITICAL INSTRUCTIONS:**
1.  **CODE ONLY:** You MUST respond with ONLY raw Python code. Do not include any explanations or markdown.
2.  **USE `df`:** The pandas DataFrame is available as the variable `df`.
3.  **FINAL RESULT:** The code MUST end with a dictionary named `result`.
    - For a DataFrame/Series, use key 'df'. Example: `result = {{'df': my_dataframe}}`
    - For a single value, use key 'text'. Example: `result = {{'text': f'The answer is {{my_value}}'}}`
    - For a chart, use key 'chart_path'. Example: `result = {{'chart_path': save_chart(plt)}}`
"""
        code = self._call_openai_api(prompt)
        cleaned_code = re.sub(r"^```(?:python)?\n|\n```$", "", code, flags=re.MULTILINE).strip()
        logger.info(f"--- Generated Code ---\n{cleaned_code}\n----------------------")

        local_scope = {}
        try:
            logger.info("Step 3: Executing code in a safe environment...")
            bytecode = compile_restricted(cleaned_code, "<inline>", "exec")
            exec(bytecode, self._globals, local_scope)
            logger.info("✅ Code executed successfully.")
            return local_scope.get("result")
        except Exception as e:
            logger.error(f"Code execution failed: {type(e).__name__}: {e}")
            raise

    def _summarize_result(self, query: str, result: Dict[str, Any]) -> str:
        """Summarizes the execution result into a final, business-friendly HTML output."""
        logger.info("Step 4: Formatting output and generating summary...")
        raw_result_html = ""
        chart_html = ""

        if not result or not isinstance(result, dict):
            raw_result_html = "<p>The analysis did not produce a valid result.</p>"
        elif 'df' in result:
            df_res = result['df']
            if isinstance(df_res, (pd.DataFrame, pd.Series)) and not df_res.empty:
                if isinstance(df_res, pd.Series): df_res = df_res.to_frame()
                raw_result_html = df_res.head(20).to_html(index=False, classes="table", border=0)
            else:
                raw_result_html = "<p>The analysis returned an empty dataset.</p>"
        elif 'text' in result:
            raw_result_html = f"<p>{result['text']}</p>"
        
        if 'chart_path' in result and result['chart_path']:
            public_url = f"/{result['chart_path'].replace(os.sep, '/')}"
            chart_html = f"<img src='{public_url}' alt='Analysis Chart' style='max-width:100%; margin-bottom:1rem;'>"

        summary_prompt = """
You are a business insights analyst. Your task is to interpret a raw analysis output and provide a clear, concise summary in HTML.
**Original User Question:** "{query}"
**Raw Analysis Result (HTML format):**
{raw_result_html}
**Instructions:**
- Provide a short executive summary that directly answers the user's question.
- If a table is present, briefly explain what it shows.
- If a chart was generated, provide a short caption for it.
- Convert technical column names to readable labels (e.g., 'chargeable_area_sft' -> 'Chargeable Area (sq ft)').
- Return **ONLY** the summary as a clean HTML block (e.g., using `<p>` and `<h4>` tags).
"""
        summary_html = self._call_openai_api(summary_prompt, temperature=0.2)
        return chart_html + summary_html + raw_result_html

    def process_query(self, query: str) -> Generator[str, None, None]:
        """
        Processes a user query through the full pipeline and yields a single, final HTML answer.
        """
        logger.info(f"Processing query: {query}")
        self.conversation_history.append({"role": "user", "content": query})

        try:
            # 1. Intent / Clarification
            intent = self._get_intent(query)
            if intent.get("type") == "clarification":
                question = intent.get("question", "Could you please clarify your request?")
                self.conversation_history.append({"role": "assistant", "content": question})
                yield question
                return

            # 2. Generate and Execute Code (No planning step)
            result = self._generate_and_execute_code(query)
            if result is None:
                raise LLMError("Code execution failed to produce a result.")

            # 3. Summarize Result and yield final answer
            final_output = self._summarize_result(query, result)
            self.conversation_history.append({"role": "assistant", "content": final_output})
            yield final_output

        except Exception as e:
            logger.error(f"Error during query processing: {e}", exc_info=True)
            error_message = f"⚠️ Sorry, I encountered an error while processing your query: {e}"
            self.conversation_history.append({"role": "assistant", "content": error_message})
            yield error_message

    def suggest_questions(self) -> List[str]:
        """Provides a list of suggested questions based on the DataFrame's columns."""
        suggestions = [
            "Which tenants have the largest deal sizes?",
            "Show a distribution of rental rates by floor.",
            "How many units are vacant versus leased?",
            "What’s the average chargeable area per unit?",
            "Plot a histogram of chargeable area."
        ]
        numeric_columns = self.df.select_dtypes(include='number').columns.tolist()
        if numeric_columns:
            col1 = numeric_columns[0]
            suggestions.append(f"What is the mean and standard deviation of {col1}?")
            if len(numeric_columns) > 1:
                col2 = numeric_columns[1]
                suggestions.append(f"Is there a correlation between {col1} and {col2}?")
        return suggestions[:8]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Returns the current conversation history."""
        return self.conversation_history

    def clear_conversation_history(self):
        """Clears the conversation history."""
        self.conversation_history.clear()
        logger.info("Cleared conversation history.")
