"""
Text-to-SQL chatbot UI (same LangChain + Groq pipeline as the notebook).

Run:  streamlit run chatbot_app.py

.env file (same folder):
    GROQ_API_KEY=your_key
    DB_PASSWORD=your_mysql_password
    # optional (defaults shown): DB_USER=root  DB_HOST=localhost  DB_PORT=3306  DB_NAME=text_to_sql
"""
import os
import re
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from sqlalchemy import create_engine, text

load_dotenv()

st.set_page_config(page_title="Text-to-SQL Chatbot", page_icon="🗄️", layout="wide")

EVAL_FILE = "eval_results.csv"  # exported from the notebook (see the export cell)
EXAMPLES = [
    "What was the budget of Product 12",
    "List all customer names from the customers table.",
    "What is the total line total for Geiss Company?",
    "What are the names of all products in the products table?",
]
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|replace|rename)\b", re.IGNORECASE
)


def is_safe(query: str) -> bool:
    """Allow a single read-only SELECT / WITH statement only."""
    q = query.strip().rstrip(";").strip()
    if not q or ";" in q:
        return False
    if not q.lower().lstrip("( ").startswith(("select", "with")):
        return False
    return not FORBIDDEN.search(q)


# ------------------------------------------------------------------ pipeline (same as notebook)
@st.cache_resource(show_spinner="Connecting to MySQL and Groq...")
def load_resources():
    user = os.getenv("DB_USER", "root")
    pwd = quote_plus(os.environ["DB_PASSWORD"])  # handles @ # / : in passwords
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "text_to_sql")

    engine = create_engine(f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}")
    db = SQLDatabase(engine, sample_rows_in_table_info=2)
    schema = db.get_table_info()  # retrieval: fetched once and reused for every question

    template = """Based on the table schema below, write a SQL query that would answer the user's question:
Remember : Only provide me the sql query dont include anything else.
           Provide me sql query in a single line dont add line breaks.
           Use backticks around table and column names that contain spaces.
Table Schema:
{schema}

Question: {question}
SQL Query:
"""
    prompt = ChatPromptTemplate.from_template(template)
    sql_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)  # reads GROQ_API_KEY

    sql_chain = (
        RunnablePassthrough.assign(schema=lambda _: schema)
        | prompt
        | sql_llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    return engine, sql_llm, sql_chain, schema


def answer(question: str) -> dict:
    """question -> generated SQL -> result table. Retries once, feeding the DB error back to the LLM."""
    ask_text, query, error, attempt = question, None, None, 0
    for attempt in (1, 2):
        resp = sql_chain.invoke({"question": ask_text})
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", resp, re.DOTALL | re.IGNORECASE)
        query = match.group(1).strip() if match else resp.strip()

        if not is_safe(query):
            return {"sql": query, "df": None, "error": "Blocked: only a single read-only SELECT query is allowed.",
                    "attempts": attempt}
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(query.rstrip(";")), conn).head(500)
            return {"sql": query, "df": df, "error": None, "attempts": attempt}
        except Exception as e:
            error = str(e).splitlines()[0][:300]
            ask_text = (
                f"{question}\n\nYour previous query failed.\nQuery: {query}\nError: {error}\n"
                "Write a corrected query."
            )
    return {"sql": query, "df": None, "error": error, "attempts": attempt}


def explain(question: str, df: pd.DataFrame) -> str:
    msg = (
        f"Question: {question}\n\nQuery result (first rows):\n{df.head(20).to_string(index=False)}\n\n"
        "Answer the question in one or two plain-English sentences using only this result."
    )
    return sql_llm.invoke(msg).content.strip()


# ------------------------------------------------------------------ startup checks
missing = [k for k in ("GROQ_API_KEY", "DB_PASSWORD") if not os.getenv(k)]
if missing:
    st.error(f"Missing in your .env file: {', '.join(missing)}")
    st.stop()

try:
    engine, sql_llm, sql_chain, schema = load_resources()
except Exception as e:
    st.error(f"Could not start: {str(e).splitlines()[0][:300]}")
    st.info("Check that MySQL is running and that DB_PASSWORD / DB_USER / DB_NAME in .env are correct.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🗄️ Text-to-SQL")
    page = st.radio("View", ["Chat", "Evaluation"], label_visibility="collapsed")
    st.divider()
    explain_on = st.toggle("Explain results in plain English", value=True)

    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state.queued = ex

    with st.expander("Database schema the LLM sees"):
        st.code(schema, language="sql")

    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()


# ------------------------------------------------------------------ pages
def render(item: dict):
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        if item.get("summary"):
            st.write(item["summary"])
        st.code(item["sql"] or "(no query generated)", language="sql")
        if item["error"]:
            st.error(f"Query failed after {item['attempts']} attempt(s): {item['error']}")
            return
        df = item["df"]
        if df.empty:
            st.info("The query ran but returned no rows.")
            return
        st.dataframe(df)
        if item["attempts"] > 1:
            st.caption("The first query failed, so it was corrected automatically.")
        # simple auto-chart: one label column + one numeric column
        if df.shape[1] == 2 and df.shape[0] > 1 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
            st.bar_chart(df.set_index(df.columns[0]))


if page == "Chat":
    st.header("Ask your database anything")
    st.caption("Question → schema lookup → LLM writes SQL → safety check → MySQL → answer")

    for item in st.session_state.history:
        render(item)

    question = st.chat_input("e.g. What is the total line total for Geiss Company?")
    question = question or st.session_state.pop("queued", None)

    if question:
        with st.spinner("Thinking..."):
            result = answer(question)
            item = {"question": question, "summary": None, **result}
            if explain_on and result["df"] is not None and not result["df"].empty:
                try:
                    item["summary"] = explain(question, result["df"])
                except Exception:
                    pass  # the table is still shown if the explanation call fails
        st.session_state.history.append(item)
        render(item)

else:
    st.header("Evaluation results")
    st.caption("Generated SQL vs hand-written reference SQL on a fixed test set (run in the notebook).")

    if not os.path.exists(EVAL_FILE):
        st.info(f"No {EVAL_FILE} found. Run the export cell at the end of the notebook, then refresh.")
    else:
        ev = pd.read_csv(EVAL_FILE)
        c1, c2, c3 = st.columns(3)
        c1.metric("Test questions", len(ev))
        if "execution_correct" in ev:
            c2.metric("Execution accuracy", f"{ev['execution_correct'].mean():.0%}")
        if "context_precision" in ev:
            c3.metric("Context precision (RAGAS)", f"{ev['context_precision'].mean():.2f}")
        st.dataframe(ev)
        st.markdown(
            "- **Execution accuracy**: the generated and reference queries return the same result on MySQL.\n"
            "- **Context precision**: RAGAS judge's score for whether the retrieved schema was relevant."
        )
