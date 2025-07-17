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
from RestrictedPython.Guards import safer_getattr, full_write_guard

# --- Configuration ---
matplotlib.use("Agg") # Use non-interactive backend for saving plots
sns.set_theme(style="whitegrid")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass

class StatisticalLLMAgent:
    """
    An intelligent agent that answers questions about a DataFrame by generating and safely executing
    Python code. It follows a structured process of intent detection, planning, execution, and summarization,
    returning only the final answer.
    """
    def __init__(self, df: pd.DataFrame, analyzer=None): # analyzer is kept for signature compatibility
        """
        Initializes the agent with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be analyzed.
            analyzer: Kept for compatibility with the original class signature, but is not used.
        """
        self._setup_openai_client()
        self.df = self._prepare_dataframe(df)
        self.dataset_context = self._generate_dataset_context()
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
        logger.info("DataFrame columns sanitized.")

        numeric_cols_to_clean = ['current_rent_chargeable_inr_per_sft', 'chargeable_area_sft']
        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"Cleaned numeric column: '{col}'")
        return df

    def _make_save_chart_function(self):
        """Creates a function to save matplotlib plots to a file."""
        def save_chart(plt_obj: matplotlib.pyplot) -> str:
            output_dir = "exports/charts"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{uuid.uuid4().hex}.png"
            path = os.path.join(output_dir, filename)
            plt_obj.savefig(path, bbox_inches="tight", dpi=150)
            plt_obj.close() # Close the plot to free up memory
            logger.info(f"Chart saved to {path}")
            return path
        return save_chart

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Creates a dictionary of globals safe for use in restricted code execution."""
        restricted_globals = safe_globals.copy()
        restricted_globals.update({
            "_getattr_": safer_getattr,
            "_write_": full_write_guard,
            "pd": pd, "np": np, "plt": plt, "sns": sns,
            "df": self.df,
            "save_chart": self._make_save_chart_function()
        })
        return restricted_globals

    def _generate_dataset_context(self) -> str:
        """Generates a string containing schema and a sample of the DataFrame."""
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        context = """
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
        """Makes a call to the OpenAI API and returns the response content."""
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except APIError as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise LLMError(f"OpenAI API call failed: {str(e)}")

    def _get_intent(self, query: str) -> Dict[str, Any]:
        """Checks if the query is clear or ambiguous."""
        intent_prompt = """
Analyze the user's query to determine if it is clear enough to be answered or if it is ambiguous.
Respond in JSON format with a 'type' and an optional 'question' if clarification is needed.

- If the query is clear, return: {{"type": "clear_query"}}
- If the query is ambiguous, return: {{"type": "clarification", "question": "Your clarifying question here."}}

User Query: "{query}"
"""
        try:
            response_str = self._call_openai_api(intent_prompt, temperature=0.0, json_mode=True)
            return json.loads(response_str)
        except (json.JSONDecodeError, LLMError) as e:
            logger.warning(f"Intent check failed: {e}. Proceeding as if query is clear.")
            return {"type": "clear_query"}

    def _get_analysis_plan(self, query: str) -> str:
        """Generates a step-by-step plan to answer the user's query."""
        prompt = """
You are an expert data analysis planner. Create a concise, step-by-step plan to answer the user's question based on the provided DataFrame context.

**DataFrame Context:**
{self.dataset_context}

**User Question:** "{query}"

**Instructions:**
Provide a clear, one-line plan. The plan should be a set of instructions for a programmer to follow. If a plot is requested, the plan should include saving the plot using the `save_chart(plt)` function.
"""
        return self._call_openai_api(prompt)

    def _generate_and_execute_code(self, plan: str) -> Union[Dict[str, Any], None]:
        """Generates and safely executes Python code based on the analysis plan."""
        logger.info("Step 2: Generating Python code...")
        prompt = """
You are an expert Python code generator for data analysis.
**Plan to Execute:** {plan}

**Instructions:**
- Write standard Python code using pandas, numpy, and matplotlib/seaborn. The DataFrame is available as `df`.
- For plots, use `plt.figure()` to create a figure, generate the plot, and pass the `plt` object to `save_chart()`.
- The final output **MUST** be a dictionary named `result`.
- If the result is a DataFrame/Series, use key 'df'. Ex: `result = {{'df': my_dataframe}}`.
- If the result is a single value, use key 'text'. Ex: `result = {{'text': f'Total: {{val}}'}}`.
- If a chart is created, use key 'chart_path'. Ex: `result = {{'chart_path': save_chart(plt)}}`.
- **DO NOT** include `import` statements or modify `df` in-place.
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
                raw_result_html = df_res.to_frame().head(20).to_html(index=False, classes="table", border=0)
            else:
                raw_result_html = "<p>The analysis returned an empty dataset.</p>"
        elif 'text' in result:
            raw_result_html = f"<p>{result['text']}</p>"
        
        if 'chart_path' in result and result['chart_path']:
            # Assuming a web server can serve files from the 'exports' directory
            public_url = f"/{result['chart_path'].replace(os.sep, '/')}"
            chart_html = f"<img src='{public_url}' alt='Analysis Chart' style='max-width:100%; margin-bottom:1rem;'>"

        summary_prompt = """
You are a business insights analyst. Your task is to interpret a raw analysis output and provide a clear, concise summary in HTML.
**Original User Question:** "{query}"
**Raw Analysis Result (HTML format):**
{raw_result_html}

**Instructions:**
- Provide a short executive summary that directly answers the user's question.
- If a table is present, briefly explain what it contains.
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
            intent = self_get_intent(query)
            if intent.get("type") == "clarification":
                question = intent.get("question", "Could you please clarify your request?")
                self.conversation_history.append({"role": "assistant", "content": question})
                yield question
                return

            # 2. Create Analysis Plan
            logger.info("Step 1: Creating an analysis plan...")
            plan = self._get_analysis_plan(query)
            logger.info(f"✅ Plan: {plan}")

            # 3. Generate and Execute Code
            result = self._generate_and_execute_code(plan)
            if result is None:
                raise LLMError("Code execution failed to produce a result.")

            # 4. Summarize Result and yield final answer
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
