import os
import uuid
import logging
import re
import argparse
from typing import Generator, Dict, Any, Union, List

# Third-party libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI, APIError
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import safer_getattr, full_write_guard

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
    An intelligent agent that answers questions about a DataFrame by generating and safely
    executing Python code, inspired by the SafeCodeAgent architecture. It includes
    planning, self-correction, and summarization steps.
    """
    def __init__(self, df: pd.DataFrame, analyzer=None):
        self._setup_openai_client()
        self.df = self._prepare_dataframe(df)
        self.actual_columns = self.df.columns.tolist()
        self.df_info = self._generate_df_info()
        self._globals = self._create_safe_globals()
        self.conversation_history: List[Dict[str, str]] = []

    def _setup_openai_client(self):
        """Sets up the OpenAI client from environment variables."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY not found in environment.")
        self.client = OpenAI(api_key=api_key)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the DataFrame by cleaning column names and handling potential type issues."""
        sanitized_columns = {col: re.sub(r'[^0-9a-zA-Z_]+', '_', col) for col in df.columns}
        df = df.rename(columns=sanitized_columns)
        logger.info(f"Original columns renamed for easier access: {sanitized_columns}")

        numeric_cols_to_clean = ['current_rent_chargeable_inr_per_sft', 'chargeable_area_sft']
        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Creates a dictionary of globals safe for use in restricted code execution."""
        restricted_globals = safe_globals.copy()
        restricted_globals.update({
            "_getattr_": safer_getattr, "_write_": full_write_guard,
            "pd": pd, "np": np, "plt": plt, "sns": sns,
            "df": self.df, "save_chart": self._make_save_chart_function()
        })
        return restricted_globals

    def _make_save_chart_function(self):
        """Creates a function to save matplotlib plots to a file."""
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

    def _generate_df_info(self) -> str:
        """Generates a string containing schema and a sample of the DataFrame."""
        info_str = "DataFrame Columns and Data Types:\n"
        for col in self.df.columns:
            info_str += f"- `{col}`: {self.df[col].dtype}\n"
        info_str += "\nSample Data (first 3 rows):\n" + self.df.head(3).to_markdown(index=False)
        return info_str

    def _call_openai_api(self, prompt: str, model: str = "gpt-4o", temperature: float = 0.0) -> str:
        """Makes a call to the OpenAI API and returns the response content."""
        try:
            response = self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMError(f"OpenAI API call failed: {str(e)}")

    def _get_analysis_plan(self, query: str) -> str:
        """Generates a step-by-step plan to answer the user's query."""
        prompt = """
You are an expert data analysis planner. Your task is to create a concise, step-by-step plan to answer a user's question.
**DataFrame Information:**
{self.df_info}
**User Question:** "{query}"
**Instructions:**
Provide a clear, one-line plan. Be smart about it. If the user asks for "first floor", your plan should involve checking for multiple variations like "1", "First", "1st", etc.
Example Plan: "Filter the DataFrame for rows where `complex` is 'Inorbit Mall (Hyderabad)' and `floor` is in a list of first-floor values, then extract the unique brand names."
"""
        return self._call_openai_api(prompt)

    def _get_corrected_plan(self, original_plan: str, error_message: str) -> str:
        """Generates a corrected plan after a code execution error."""
        logger.info("Revisiting the plan due to a data or plan error...")
        prompt = """
You are an expert data analysis plan corrector. A plan failed to execute. Your task is to create a new, safer plan.
**Original Plan:** {original_plan}
**Execution Error:** {error_message}
**Available DataFrame Columns:** {self.actual_columns}
**Instructions:**
Analyze the error. If it's a `ValueError` or `TypeError`, the data in a column is likely dirty. The new plan MUST focus on cleaning that column. If it's a `KeyError`, a column name is wrong. Correct the plan to use a valid column name. Provide only the new, single-line plan.
"""
        corrected_plan = self._call_openai_api(prompt)
        logger.info(f"✅ Corrected Plan: {corrected_plan}")
        return corrected_plan

    def _generate_and_execute_code(self, plan: str, max_retries: int) -> Union[Dict[str, Any], None]:
        """Generates and executes code, with a retry mechanism for correctable errors."""
        current_plan = plan
        for attempt in range(max_retries):
            logger.info(f"Step 2: Generating Python code (Attempt {attempt + 1}/{max_retries})...")
            prompt = """
You are an expert Python code generator for data analysis.
**Plan to Execute:** {current_plan}
**Instructions:**
- Write standard Python code using pandas. The DataFrame is available as `df`.
- **HANDLE MESSY DATA:** The `floor` column might contain comma-separated values like "Ground,1,2". To check if "1" is on that floor, you must use `str.contains('1')`. A simple `==` check will fail.
- **CRITICAL SAFETY RULE:** You **MUST NOT** modify the `df` DataFrame.
- **RESULT FORMAT:** The final output **MUST** be a dictionary named `result`.
    - For a DataFrame/Series, use key 'df'. Example: `result = {{'df': my_dataframe}}`.
    - For a single value, use key 'text'. Example: `result = {{'text': f'The answer is {{my_value}}'}}`.
    - For a chart, use key 'chart_path'. Example: `result = {{'chart_path': save_chart(plt)}}`.
- **DO NOT** include any `import` statements.
"""
            code = self._call_openai_api(prompt)
            cleaned_code = re.sub(r"^```(?:python)?\n|\n```$", "", code, flags=re.MULTILINE).strip()
            logger.info(f"--- Generated Code ---\n{cleaned_code}\n----------------------")
            
            local_scope = {}
            try:
                logger.info("Step 3: Executing code...")
                bytecode = compile_restricted(cleaned_code, "<inline>", "exec")
                exec(bytecode, self._globals, local_scope)
                logger.info("✅ Code executed successfully.")
                return local_scope.get("result")
            except Exception as e:
                error_str = f"{type(e).__name__}: {e}"
                logger.error(f"❌ Execution failed: {error_str}")
                
                if isinstance(e, (KeyError, ValueError, TypeError)) and attempt < max_retries - 1:
                    current_plan = self._get_corrected_plan(current_plan, error_str)
                    continue
                else:
                    logger.error("Max retries reached or uncorrectable error. Halting execution.")
                    raise e
        
        raise LLMError(f"Code generation failed after {max_retries} attempts.")

    def _summarize_result(self, query: str, plan: str, result: Dict[str, Any]) -> str:
        """Summarizes the execution result into a final, business-friendly HTML output."""
        logger.info("Step 4: Formatting output and generating summary...")
        output_html = ""
        result_data = result.get('df')

        if isinstance(result_data, pd.DataFrame):
            output_html = result_data.to_html(index=False, classes="table table-striped", border=0)
        elif isinstance(result_data, pd.Series):
            output_html = result_data.to_frame().to_html(index=False, classes="table table-striped", border=0)
        elif "text" in result:
            output_html = f'<p style="font-size: 1.1em;">{result["text"]}</p>'
        elif "chart_path" in result:
            relative_path = os.path.relpath(result["chart_path"]).replace("\\", "/")
            output_html = f"<img src='/{relative_path}' alt='Generated Chart' style='max-width:100%; height:auto;'/>"

        summary_prompt = """
You are a business analyst. Your goal is to provide a clear, concise summary of a data analysis result for a non-technical audience.
**Original User Question:** "{query}"
**Final Analysis Plan:** "{plan}"
**Result Data (HTML format):**
{output_html}
**Instructions:**
Start with a direct answer to the user's question. If there is a table, mention what it contains. If the result is text, summarize its findings. If the result is empty or NaN, state that the requested data could not be found. Keep the summary professional.
"""
        summary = self._call_openai_api(summary_prompt, temperature=0.3)
        
        return f'<div><p><strong>Summary</strong></p><p>{summary}</p></div><br>{output_html}'

    def process_query(self, query: str, max_retries: int = 3) -> Generator[str, None, None]:
        """Processes a user query and yields a single, final HTML answer."""
        self.conversation_history.append({"role": "user", "content": query})
        try:
            logger.info("Step 1: Creating an analysis plan...")
            plan = self._get_analysis_plan(query)
            if "cannot be answered" in plan.lower():
                yield f"⚠️ **Analysis Stopped:** {plan}"
                return
            logger.info(f"✅ Initial Plan: {plan}")

            result = self._generate_and_execute_code(plan, max_retries)
            
            if result is None:
                raise LLMError("Could not retrieve a result from code execution.")

            final_output = self._summarize_result(query, plan, result)
            self.conversation_history.append({"role": "assistant", "content": final_output})
            yield final_output

        except Exception as e:
            logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
            error_message = f"⚠️ An unexpected error occurred: {e}"
            self.conversation_history.append({"role": "assistant", "content": error_message})
            yield error_message

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Returns the current conversation history."""
        return self.conversation_history

    def clear_conversation_history(self):
        """Clears the conversation history."""
        self.conversation_history.clear()
        logger.info("Cleared conversation history.")

    def suggest_questions(self) -> List[str]:
        """Provides a list of suggested questions based on the DataFrame's columns."""
        suggestions = [
            "Which tenants have the largest deal sizes?",
            "Show a distribution of rental rates.",
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
