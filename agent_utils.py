import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import torch

# -----------------------------
# Database connection
# -----------------------------
engine = create_engine("sqlite:///ecommerce.db")

# -----------------------------
# Models
# -----------------------------
SQL_MODEL_NAME = "defog/sqlcoder-7b-2"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(SQL_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    SQL_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

# -----------------------------
# Load DB schema
# -----------------------------
def load_schema():
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()

    schema = []
    for (table,) in tables:
        with engine.connect() as conn:
            cols = conn.execute(
                text(f"PRAGMA table_info({table})")
            ).fetchall()
            col_names = ", ".join([c[1] for c in cols])
            schema.append(f"{table}: {col_names}")

    return schema

SCHEMA_TEXTS = load_schema()
schema_embeddings = embedder.encode(SCHEMA_TEXTS, convert_to_numpy=True)

index = faiss.IndexFlatL2(schema_embeddings.shape[1])
index.add(schema_embeddings)

def retrieve_schema(question, top_k=3):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    _, idx = index.search(q_emb, top_k)
    return "\n".join([SCHEMA_TEXTS[i] for i in idx[0]])

# -----------------------------
# Text → SQL
# -----------------------------
def generate_sql(question, memory_context="", top_k=3):
    schema = retrieve_schema(question, top_k)

    prompt = f"""
You are an expert SQLite SQL generator.

Database schema:
{schema}

Conversation context:
{memory_context}

Question:
{question}

Return ONLY a valid SQLite SQL query.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql.split("```")[-1].strip()

# -----------------------------
# SQL execution + auto-repair
# -----------------------------
def run_sql(sql):
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)

def run_sql_with_autorepair(question, sql, max_retries=2):
    last_error = None

    for _ in range(max_retries + 1):
        try:
            df = run_sql(sql)
            return sql, df
        except Exception as e:
            last_error = str(e)
            sql = generate_sql(
                f"{question}\nFix this SQLite error:\n{last_error}"
            )

    raise RuntimeError(last_error)

# -----------------------------
# Explanation
# -----------------------------
def explain_results(question, sql, df):
    return f"Results for **{question}** based on executed SQL."

# -----------------------------
# Auto-plot
# -----------------------------
def auto_plot(df, title=""):
    if df.shape[1] >= 2:
        fig, ax = plt.subplots()
        df.iloc[:, :2].plot(kind="bar", ax=ax)
        ax.set_title(title)
        return fig
    return None
