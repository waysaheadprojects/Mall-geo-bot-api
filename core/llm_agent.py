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
        # --- FIX STARTS HERE ---
        # Clean potentially problematic numeric columns upfront.
        # This prevents the TypeError before it can happen in pandasai.
        numeric_cols_to_clean = ['current_rent_chargeable_inr_per_sft', 'chargeable_area_sft']
        for col in numeric_cols_to_clean:
            if col in self.df.columns:
                # Use pd.to_numeric to convert the column, coercing errors to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        # --- FIX ENDS HERE ---

        self.analyzer = analyzer
        self.conversation_history: List[Dict[str, str]] = []

        self.llm_wrapper = OpenAILLM()
        self.client = self.llm_wrapper.client

        self.pandasai_llm = PandasAIOpenAI(api_token=os.environ.get("OPENAI_API_KEY"))
        self.smart_dataframe = SmartDataframe(self.df, config={ # Use the cleaned self.df
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
            """
            Run a user query through PandasAI, then rephrase the raw analytical result into
            a business-friendly, HTML-formatted answer (no code shown).
            """
            logger.info(f"Processing query: {query}")
        
            try:
                # ------------------------------------------------------------------
                # 1. Intent / clarification
                # ------------------------------------------------------------------
                intent = self._get_intent(query)
                if intent.get("type") == "clarification":
                    question = intent.get("question", "Could you clarify your query?")
                    self.conversation_history.append({"role": "assistant", "content": question})
                    yield question
                    return
        
                # ------------------------------------------------------------------
                # 2. Run PandasAI
                # ------------------------------------------------------------------
                self.conversation_history.append({"role": "user", "content": query})
                full_prompt = self._build_prompt(query)
                pandasai_response = self.smart_dataframe.chat(full_prompt)
        
                # ------------------------------------------------------------------
                # 3. Normalize PandasAI output
                #    We try to extract: (result_df, scalar, text, chart_path)
                # ------------------------------------------------------------------
                result_df = None
                scalar_value = None
                text_blob = None
                chart_path = None
        
                if isinstance(pandasai_response, pd.DataFrame):
                    result_df = pandasai_response
                elif isinstance(pandasai_response, str):
                    stripped = pandasai_response.strip()
                    if stripped.endswith(".png") and os.path.exists(stripped):
                        chart_path = stripped
                    else:
                        text_blob = stripped
                elif isinstance(pandasai_response, (int, float, bool)):
                    scalar_value = pandasai_response
                elif isinstance(pandasai_response, dict):
                    ptype = pandasai_response.get("type")
                    pval = pandasai_response.get("value")
                    if ptype == "dataframe" and isinstance(pval, pd.DataFrame):
                        result_df = pval
                    elif ptype == "plot" and isinstance(pval, str) and pval.endswith(".png"):
                        if os.path.exists(pval):
                            chart_path = pval
                        else:
                            text_blob = str(pval)
                    else:
                        text_blob = str(pandasai_response)
                elif isinstance(pandasai_response, (list, tuple)):
                    for item in pandasai_response:
                        if isinstance(item, pd.DataFrame) and result_df is None:
                            result_df = item
                        elif isinstance(item, dict):
                            itype = item.get("type")
                            ival = item.get("value")
                            if itype == "dataframe" and isinstance(ival, pd.DataFrame) and result_df is None:
                                result_df = ival
                            elif itype == "plot" and isinstance(ival, str) and ival.endswith(".png"):
                                if os.path.exists(ival):
                                    chart_path = ival
                            else:
                                text_blob = str(item)
                        elif isinstance(item, str) and item.endswith(".png") and os.path.exists(item):
                            chart_path = item
                        elif isinstance(item, str):
                            text_blob = item
                        elif isinstance(item, (int, float, bool)) and scalar_value is None:
                            scalar_value = item
                else:
                    text_blob = str(pandasai_response)
        
                # ------------------------------------------------------------------
                # 4. Build a structured "analysis result" string to feed the chat model
                # ------------------------------------------------------------------
                parts = []
                ctx = self.dataset_context
                if len(ctx) > 1200:
                    ctx = ctx[:1200] + "\n...[truncated]..."
                parts.append("DATASET CONTEXT (truncated):\n" + ctx)
                parts.append(f"USER QUERY:\n{query}")
        
                if result_df is not None:
                    parts.append(
                        "RESULT DATAFRAME PREVIEW (first 10 rows):\n" +
                        result_df.head(10).to_markdown(index=False)
                    )
                    parts.append(f"RESULT DATAFRAME SHAPE: {result_df.shape[0]} rows x {result_df.shape[1]} columns")
                if scalar_value is not None:
                    parts.append(f"SCALAR RESULT: {scalar_value}")
                if text_blob:
                    parts.append(f"PANDASAI TEXT RESULT:\n{text_blob}")
                if chart_path:
                    parts.append(f"CHART PATH: {chart_path}")
        
                result_str = "\n\n".join(parts)
        
                # ------------------------------------------------------------------
                # 5. Build GPT prompt for business explanation
                # ------------------------------------------------------------------
                gpt_prompt = f"""
        You are a business insights analyst.
        Your task is to interpret raw analysis output in business language.
        1. Provide a short executive summary.
        2. If numbers are present, state the key metrics clearly.
        3. If a table is needed, output an HTML table.
        4. If a chart is mentioned, include a short caption for it.
        5. Convert technical column names to readable labels (e.g., 'chargeable_area_sft' -> 'Chargeable Area (sq ft)').
        Return ONLY HTML.
        --- RAW ANALYSIS BELOW ---
        {result_str}
        """
        
                # ------------------------------------------------------------------
                # 6. Call GPT to convert raw result -> business HTML
                # ------------------------------------------------------------------
                gpt_response = self.client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[{"role": "user", "content": gpt_prompt}],
                    temperature=0.3,
                    max_tokens=1200,
                )
                business_html = gpt_response.choices[0].message.content.strip()
        
                # ------------------------------------------------------------------
                # 7. Inject chart <img> (if any) ahead of explanation
                # ------------------------------------------------------------------
                chart_html = ""
                if chart_path:
                    chart_path_for_url = chart_path.replace(os.sep, "/")
                    public_url = f"https://mallgpt.waysaheadglobal.com/{chart_path_for_url}"
                    chart_html = f"<img src='{public_url}' alt='Analysis Chart' style='max-width:100%; margin-bottom:1rem;'>"
        
                final_output = chart_html + business_html
        
                # ------------------------------------------------------------------
                # 8. Save to history + return
                # ------------------------------------------------------------------
                self.conversation_history.append({"role": "assistant", "content": final_output} )
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
