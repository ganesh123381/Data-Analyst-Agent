from dotenv import load_dotenv
import os
load_dotenv()

import pandas as pd
from langchain_ollama import ChatOllama


# ─────────────────────────────────────────
# Clean Data
# ─────────────────────────────────────────
def clean_data(df):
    df = df.drop_duplicates()

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    return df


# ─────────────────────────────────────────
# Generate Auto Report
# ─────────────────────────────────────────
def generate_report(df):
    report = []

    report.append(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    report.append(f"**Columns:** {', '.join(df.columns.tolist())}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    report.append(
        f"**Missing Values:**\n{missing.to_string() if not missing.empty else 'None'}"
    )

    report.append(f"**Numeric Summary:**\n{df.describe().to_string()}")

    return "\n\n".join(report)


# ─────────────────────────────────────────
# Smart Hybrid Agent (Stable Production Version)
# ─────────────────────────────────────────
def run_agent(df, question, chat_history=None):

    q = question.lower()

    # ⚡ Fast built-in responses (no LLM needed)
    if "number of rows" in q or "total rows" in q:
        return f"The dataset contains **{df.shape[0]:,} rows**."

    if "number of columns" in q or "total columns" in q:
        return f"The dataset contains **{df.shape[1]} columns**."

    if "duplicate" in q:
        return f"There are **{df.duplicated().sum()} duplicate rows**."

    try:
        llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0,
        base_url="http://localhost:11434"
)

        # STEP 1: Generate Python computation
        code_prompt = f"""
You are a Python data analyst.

The dataframe is named df.
Columns: {df.columns.tolist()}

User question:
{question}

Return ONLY valid Python code.
Store final useful output in a variable named result.
Do not print anything.
"""

        response = llm.invoke(code_prompt)
        code = response.content.strip()

        # Remove markdown formatting if present
        code = code.replace("```python", "").replace("```", "")

        local_vars = {"df": df}
        exec(code, {}, local_vars)

        result = local_vars.get("result", None)

        # STEP 2: If result is boolean or empty → explain clearly
        if isinstance(result, bool) or result is None:
            explain_prompt = f"""
The user asked:
"{question}"

Provide a direct and concise analytical explanation.
Use dataset reasoning.
Do not give a general dataset summary.
Focus only on answering the question clearly.
"""
            explanation = llm.invoke(explain_prompt)
            return explanation.content.strip()

        # STEP 3: Explain computed numeric result clearly
        explain_prompt = f"""
User asked: "{question}"

Here is the computed result from the dataset:
{result}

Provide a direct and concise explanation.
Explain clearly how this was determined.
Include specific numbers from the result.
Do NOT give a general dataset summary.
Keep the explanation focused on the question only.
"""

        explanation = llm.invoke(explain_prompt)
        return explanation.content.strip()

    except Exception as e:
        return f"⚠️ Error processing question: {str(e)}"