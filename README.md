# AI Data Analyst Agent

An end-to-end GenAI app that converts natural language questions into SQL using schema-aware RAG, executes queries safely, auto-repairs SQL errors, and visualizes results in a Streamlit UI.

## Features
- Text-to-SQL using SQLCoder
- Schema RAG with embeddings + FAISS
- SQL auto-repair loop using execution errors
- Conversational follow-ups with memory
- Auto chart generation
- Streamlit chat UI

## Architecture
User Question → Schema Retrieval (FAISS) → SQL Generation → Safety Check → Execute → Auto-Repair → Results + Chart

## Tech Stack
Python, Streamlit, SQLite, SQLAlchemy, Hugging Face Transformers, SQLCoder, SentenceTransformers, FAISS, Matplotlib

## Run Locally
```bash
git clone https://github.com/sathwik7559/ai-data-analyst-agent.git
cd ai-data-analyst-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
