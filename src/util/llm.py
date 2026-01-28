
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import streamlit as st
from config import LLM_MODEL, EMBEDDING_MODEL, top_K, sqlite_table
from langchain_ollama import OllamaEmbeddings
import sqlite3
import re
from tabulate import tabulate

class GraphState(TypedDict):
    question: str
    intent: Optional[str]
    result: Optional[Any]

router_prompt = PromptTemplate.from_template("""
    Classify the following question into one of the categories below:

    general: General or high-level questions that are informational in nature and not directly tied to fraud documents or historical fraud data.

    fraud_doc: Questions specifically related to fraud documentation, including explanations of fraud types, fraud techniques, prevention methods, impacts of fraud, or conceptual discussions about credit card fraud (e.g., card-related frauds, merchant frauds, internet frauds, fraud prevention technologies).

    fraud_history: Questions related to historical fraud data, fraud records, trends, statistics, or analysis that would typically require querying databases or using SQL.

    Answer using only one of the labels above.

    Question:
    {question}
    """)

def init_llm(url):
    return ChatOllama(model=LLM_MODEL, base_url=url), OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=url)

def router_node(state: GraphState):
    response = st.session_state.llm.invoke(
        router_prompt.format(question=state["question"])
    )
    
    print(response.content.strip())
    return {
        "intent": response.content.strip()
    }
    
    
def general_llm_node(state: GraphState):
    return {
        "result": {
            "stream": st.session_state.llm.stream(state["question"]),
            "references": None
        }
    }
    
def fraud_doc_node(state: GraphState):
    
    query_prompt = f"""
    You are a rephrasing agent.
    Rewrite the user's input into a clear, well-structured, and semantically rich query that maximizes similarity matching in a vector database.

    Preserve the original intent and meaning, but improve clarity and wording.
    Do not add new information or change the intent.

    User input:
    {state["question"]}
    """
    
    query = st.session_state.llm.invoke(query_prompt).content
    
    docs = st.session_state.vectordb.similarity_search(query, k=top_K)

    context = "\n\n".join([
    f"--- Result {i+1} ---\n{doc.page_content}"
    for i, doc in enumerate(docs)])

    prompt = f"""
    Based on the following documents, answer the question and include a brief reference.

    Documents:
    {context}

    Question:
    {state["question"]}

    Additional instruction:
    Provide a concise summary with a maximum length of 200 words.
    """

    return {
        "result": {
            "stream": st.session_state.llm.stream(prompt),
            "references": f"{context} \n\n Summary : "
        }
    }

def extract_sql(text):
    pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def fraud_history_node(state: GraphState):
    conn = sqlite3.connect(st.session_state.db_path)

    try:
        cursor = conn.cursor()

        # ---- Fetch table DDL safely ----
        cursor.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (sqlite_table,)
        )

        row = cursor.fetchone()
        if row is None or row[0] is None:
            raise ValueError(f"Table '{sqlite_table}' does not exist")

        ddl = row[0]

        # ---- Build SQL generation prompt ----
        sql_prompt = f"""
        You are an expert SQL generator.
        Your task is to write correct, efficient, and readable SQL queries
        based strictly on the provided database schema and the user request.

        Rules:
        - Use ONLY the tables and columns defined in the DDL
        - Do NOT hallucinate tables or columns
        - Use clear table aliases
        - Only generate SELECT queries
        - Always include a LIMIT (max 100 rows)
        - Return ONLY the SQL query

        DDL:
        {ddl}

        User request:
        {state["question"]}
        """

        # ---- Generate SQL ----
        sql_query = st.session_state.llm.invoke(sql_prompt).content
        sql = extract_sql(sql_query).strip()

        # ---- Hard safety check ----
        normalized_sql = sql.lower()
        if not normalized_sql.startswith("select"):
            raise ValueError("Only SELECT queries are allowed")

        if "limit" not in normalized_sql:
            sql += " LIMIT 100"

        # ---- Execute SQL ----
        cursor.execute(sql)

        if cursor.description is None:
            raise ValueError("Query returned no columns")

        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchmany(100)

        sql_result = tabulate(
            rows,
            headers=columns,
            tablefmt="psql"
        )

        # ---- Summarization ----
        summary_prompt = f"""
        Based on the following table:

        {sql_result}

        Provide a concise summary (max 200 words).
        """

        return {
            "result": {
                "references": f"{sql_result}\n\nSummary:",
                "stream": st.session_state.llm.stream(summary_prompt),
            }
        }

    finally:
        conn.close()


graph = StateGraph(GraphState)

graph.add_node("router", router_node)
graph.add_node("general", general_llm_node)
graph.add_node("fraud_doc", fraud_doc_node)
graph.add_node("fraud_history", fraud_history_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["intent"],
    {
        "general": "general",
        "fraud_doc": "fraud_doc",
        "fraud_history": "fraud_history"
    }
)

graph.add_edge("general", END)
graph.add_edge("fraud_doc", END)
graph.add_edge("fraud_history", END)

app = graph.compile()